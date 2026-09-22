import pytest
import asyncio
from src.pipeline.interfaces import VADEvent, TranscriptEvent

class MockVAD:
    # Emits a configurable sequence of VAD events with delays

    def __init__(self, events: list[tuple[float, VADEvent]]) -> None:
        self._events = events

    async def stream(self, audio_in):
        for delay, event in self._events:
            await asyncio.sleep(delay)
            yield event


class MockSTT:
    def __init__(self, transcripts: list[TranscriptEvent]) -> None:
        self._transcripts = transcripts

    async def stream(self, audio_in):
        for t in self._transcripts:
            yield t


class MockLLM:
    def __init__(self, tokens: list[str], delay: float = 0.0) -> None:
        self._tokens = tokens
        self._delay = delay

    async def stream(self, messages):
        for tok in self._tokens:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield tok


class MockTTS:
    def __init__(self) -> None:
        self.synthesized_texts: list[str] = []

    async def stream(self, text: str):
        self.synthesized_texts.append(text)
        yield f"<audio:{text}>".encode()


@pytest.fixture
def mock_tts() -> MockTTS:
    return MockTTS()