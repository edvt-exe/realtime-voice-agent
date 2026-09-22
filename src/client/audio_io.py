from __future__ import annotations
import asyncio
import queue
import threading

import pyaudio
import structlog

log = structlog.get_logger()

_CHUNK_SIZE = 1024
_FORMAT = pyaudio.paInt16
_CHANNELS = 1

class MicrophoneCapture:
    # Captures raw PCM audio from the default microphone on a background thread, exposing chunks through an asyncio-friendly interface

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._pa = pyaudio.PyAudio()
        self._raw_q: queue.Queue[bytes] = queue.Queue()
        self._stream: pyaudio.Stream | None = None
        self._running = False

    def _callback(self, in_data, frame_count, time_info, status) -> tuple:
        self._raw_q.put(in_data)
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._running = True
        self._stream = self._pa.open(format=_FORMAT, channels=_CHANNELS, rate=self.sample_rate, input=True, frames_per_buffer=_CHUNK_SIZE, stream_callback=self._callback)
        self._stream.start_stream()
        log.info("microphone_started", sample_rate=self.sample_rate)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        log.info("microphone_stopped")

    async def chunks(self):
        # Async generator yielding raw PCM chunks as they are captured
        loop = asyncio.get_event_loop()
        while self._running:
            chunk = await loop.run_in_executor(None, self._raw_q.get)
            yield chunk


class SpeakerPlayback:
    # Plays back raw decoded audio chunks through the default speaker, consuming from an asyncio queue on a background thread

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._pa = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._play_q: queue.Queue[bytes | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stream = self._pa.open(format=_FORMAT, channels=_CHANNELS, rate=self.sample_rate, output=True)
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        log.info("speaker_started", sample_rate=self.sample_rate)

    def _playback_loop(self) -> None:
        while True:
            chunk = self._play_q.get()
            if chunk is None:
                break
            self._stream.write(chunk)

    def enqueue(self, chunk: bytes) -> None:
        self._play_q.put(chunk)

    def clear_queue(self) -> None:
        # Drops all pending audio — called on barge in to stop playback instantly
        while not self._play_q.empty():
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._play_q.put(None)
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        log.info("speaker_stopped")