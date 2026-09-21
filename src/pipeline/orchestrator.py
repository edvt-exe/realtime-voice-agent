from __future__ import annotations
import asyncio
import structlog

from src.config import settings
from src.pipeline.interfaces import VADProvider, STTProvider, LLMProvider, TTSProvider, VADEvent
from src.pipeline.sentence_splitter import IncrementalSentenceSplitter

log = structlog.get_logger()

class VoiceAgentOrchestrator:
    # Owns one session s full duplex pipeline: VAD -> STT -> LLM -> TTS

    def __init__(self, vad: VADProvider, stt: STTProvider, llm: LLMProvider, tts: TTSProvider) -> None:
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts

        self.conversation_history: list[dict[str, str]] = []
        self.audio_in_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=settings.queue_maxsize)
        self.audio_out_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=settings.queue_maxsize)
        self._current_gen_task: asyncio.Task | None = None

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._stt_loop(), name="stt_loop")
            tg.create_task(self._vad_loop(), name="vad_loop")

    async def push_audio_chunk(self, chunk: bytes) -> None:
        await self.audio_in_q.put(chunk)

    async def _vad_loop(self) -> None:
        async for event in self.vad.stream(self.audio_in_q):
            if event.type == "speech_start":
                await self._handle_barge_in()

    async def _handle_barge_in(self) -> None:
        task = self._current_gen_task
        if task and not task.done():
            task.cancel()
            log.info("barge_in_cancelled_generation")

        while not self.audio_out_q.empty():
            try:
                self.audio_out_q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _stt_loop(self) -> None:
        async for transcript in self.stt.stream(self.audio_in_q):
            if not transcript.is_final:
                continue
            log.info("stt_final", text=transcript.text)
            self._current_gen_task = asyncio.create_task(self._generate_response(transcript.text))

    async def _generate_response(self, user_text: str) -> None:
        self.conversation_history.append({"role": "user", "content": user_text})
        splitter = IncrementalSentenceSplitter()
        assistant_text = ""

        try:
            async for token in self.llm.stream(self.conversation_history):
                assistant_text += token
                for sentence in splitter.feed(token):
                    await self._synthesize_and_enqueue(sentence)

            remainder = splitter.flush()
            if remainder:
                await self._synthesize_and_enqueue(remainder)

            self.conversation_history.append({"role": "assistant", "content": assistant_text})
            await self.audio_out_q.put(None)

        except asyncio.CancelledError:
            log.info("generation_cancelled", partial_text=assistant_text)
            raise

    async def _synthesize_and_enqueue(self, sentence: str) -> None:
        async for audio_chunk in self.tts.stream(sentence):
            await self.audio_out_q.put(audio_chunk)