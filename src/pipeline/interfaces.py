from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class VADEvent:
    type: str
    timestamp: float


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool


class VADProvider(Protocol):
    def stream(self, audio_in: "asyncio.Queue[bytes]") -> AsyncIterator[VADEvent]: ...


class STTProvider(Protocol):
    def stream(self, audio_in: "asyncio.Queue[bytes]") -> AsyncIterator[TranscriptEvent]: ...


class LLMProvider(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class TTSProvider(Protocol):
    def stream(self, text: str) -> AsyncIterator[bytes]: ...