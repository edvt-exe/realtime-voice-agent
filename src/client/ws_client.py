from __future__ import annotations
import asyncio
import structlog
import websockets
from websockets.asyncio.client import ClientConnection

from src.client.audio_io import MicrophoneCapture, SpeakerPlayback

log = structlog.get_logger()

class VoiceClient:
    # Connects to the Echo WebSocket server, streams microphone audio out, plays back synthesized audio as it arrives and runs until interrupted
    def __init__(self, server_url: str, sample_rate: int = 16000) -> None:
        self.server_url = server_url
        self.mic = MicrophoneCapture(sample_rate=sample_rate)
        self.speaker = SpeakerPlayback(sample_rate=44100)
        self._ws: ClientConnection | None = None

    async def run(self) -> None:
        self.mic.start()
        self.speaker.start()

        try:
            async with websockets.connect(self.server_url, max_size=None) as ws:
                self._ws = ws
                log.info("connected_to_server", url=self.server_url)

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._send_loop(ws), name="send_loop")
                    tg.create_task(self._receive_loop(ws), name="receive_loop")

        except websockets.ConnectionClosed:
            log.info("disconnected_from_server")
        finally:
            self.mic.stop()
            self.speaker.stop()

    async def _send_loop(self, ws: ClientConnection) -> None:
        # Streams captured microphone chunks to the server
        async for chunk in self.mic.chunks():
            await ws.send(chunk)

    async def _receive_loop(self, ws: ClientConnection) -> None:
        # Receives synthesized audio chunks from the server and queues them for playback
        async for message in ws:
            if isinstance(message, bytes):
                self.speaker.enqueue(message)
            else:
                log.warning("unexpected_text_message", message=message)