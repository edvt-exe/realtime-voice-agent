from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator

import numpy as np
import structlog
from silero_vad import load_silero_vad, VADIterator

from src.config import settings
from src.pipeline.interfaces import VADEvent

log = structlog.get_logger()

_WINDOW_SAMPLES = 512


class SileroVAD:
    # Wraps Silero VAD for streaming speech_start/speech_end detection
    # Runs fully locally, CPU-only inference offloaded to a thread executor so it never blocks the asyncio event loop

    def __init__(self, threshold: float | None = None, min_silence_ms: int | None = None) -> None:
        self.sample_rate = settings.stt_sample_rate
        model = load_silero_vad()
        self._iterator = VADIterator(
            model,
            threshold=threshold if threshold is not None else settings.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_ms if min_silence_ms is not None else settings.vad_min_silence_ms,
        )
        self._byte_buffer = bytearray()

    async def stream(self, audio_in: "asyncio.Queue[bytes]") -> AsyncIterator[VADEvent]:
        # Consumes raw int16 PCM chunks from audio_in, yields VAD events as they fire
        loop = asyncio.get_event_loop()
        while True:
            chunk = await audio_in.get()
            self._byte_buffer.extend(chunk)

            # Silero VAD requires fixed size windows
            while len(self._byte_buffer) >= _WINDOW_SAMPLES * 2:
                window_bytes = bytes(self._byte_buffer[: _WINDOW_SAMPLES * 2])
                del self._byte_buffer[: _WINDOW_SAMPLES * 2]

                result = await loop.run_in_executor(None, self._infer, window_bytes)
                if result is None:
                    continue
                if "start" in result:
                    log.info("vad_speech_start")
                    yield VADEvent(type="speech_start", timestamp=time.monotonic())
                elif "end" in result:
                    log.info("vad_speech_end")
                    yield VADEvent(type="speech_end", timestamp=time.monotonic())

    def _infer(self, window_bytes: bytes) -> dict | None:
        pcm = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self._iterator(pcm, return_seconds=False)

    def reset(self) -> None:
        # Clears the model s internal recurrent state. Call between sessions/turns
        self._iterator.reset_states()