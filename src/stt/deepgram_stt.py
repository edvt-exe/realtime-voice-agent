from __future__ import annotations
import asyncio
from typing import AsyncIterator

import structlog
from deepgram import (DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions)

from src.config import settings
from src.pipeline.interfaces import TranscriptEvent

log = structlog.get_logger()


class DeepgramSTT:
    #Streams raw PCM audio to Deepgram s live transcription WebSocket, yields interim and final transcripts as they arrive

    def __init__(self, model: str | None = None, sample_rate: int | None = None) -> None:
        self.model = model or settings.stt_model
        self.sample_rate = sample_rate or settings.stt_sample_rate
        self.language = settings.stt_language

        client_options = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(settings.deepgram_api_key, client_options)

    async def stream(self, audio_in: "asyncio.Queue[bytes]") -> AsyncIterator[TranscriptEvent]:
        events_q: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        connection = self._client.listen.asyncwebsocket.v("1")

        async def on_message(_, result, **kwargs) -> None:
            transcript = result.channel.alternatives[0].transcript
            if not transcript:
                return
            await events_q.put(
                TranscriptEvent(text=transcript, is_final=result.is_final)
            )

        async def on_error(_, error, **kwargs) -> None:
            log.error("deepgram_error", error=str(error))

        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model=self.model,
            language=self.language,
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            endpointing=False,
            smart_format=True,
        )

        started = await connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram connection")

        sender_task = asyncio.create_task(self._forward_audio(connection, audio_in))

        try:
            while True:
                event = await events_q.get()
                yield event
        finally:
            sender_task.cancel()
            await connection.finish()

    async def _forward_audio(self, connection, audio_in: "asyncio.Queue[bytes]") -> None:
        # Continuously forwards raw PCM chunks from our internal queue to Deepgram
        while True:
            chunk = await audio_in.get()
            await connection.send(chunk)