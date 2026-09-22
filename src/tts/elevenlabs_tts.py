from __future__ import annotations
import asyncio
from typing import AsyncIterator

import structlog
from elevenlabs.client import AsyncElevenLabs

from src.config import settings

log = structlog.get_logger()

class ElevenLabsTTS:
    #Streams synthesized audio from ElevenLabs for a single piece of text
    # Each call to stream() opens its own short-lived request — sentence level granularity means we don t need a persistent bidirectional WS connectionlike Deepgram s
    # ElevenLabs streaming endpoint returns audio chunks as they are generated for one text input

    def __init__(self, voice_id: str | None = None, model: str | None = None) -> None:
        self.voice_id = voice_id or settings.tts_voice_id
        self.model = model or settings.tts_model
        self._client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        # Yields raw audio chunks as they are generated
        if not text.strip():
            return

        try:
            audio_stream = self._client.text_to_speech.stream(voice_id=self.voice_id, text=text, model_id=self.model, output_format="mp3_44100_128", optimize_streaming_latency=4)
            async for chunk in audio_stream:
                if chunk:
                    yield chunk

        except asyncio.CancelledError:
            log.info("tts_stream_cancelled")
            raise
        except Exception as e:
            log.error("elevenlabs_stream_error", error=str(e))
            raise