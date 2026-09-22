from __future__ import annotations
import asyncio
from typing import AsyncIterator

import structlog
from groq import AsyncGroq

from src.config import settings

log = structlog.get_logger()

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep responses short and "
    "conversational — you are speaking out loud, not writing text. Avoid "
    "lists, markdown, or long explanations unless explicitly asked."
)

class GroqLLM:
    # Streams chat completion tokens from Groq s OpenAI compatible API

    def __init__(self, model: str | None = None, system_prompt: str | None = None) -> None:
        self.model = model or settings.llm_model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        # messages: conversation history as [{"role": "user"/"assistant", "content": ...}, ...]
        # Yields text token-by-token (actually chunk-by-chunk, as Groq streams them)
        full_messages = [{"role": "system", "content": self.system_prompt}, *messages]
        try:
            response = await self._client.chat.completions.create(model=self.model, messages=full_messages, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature, stream=True)
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except asyncio.CancelledError:
            log.info("llm_stream_cancelled")
            raise
        except Exception as e:
            log.error("groq_stream_error", error=str(e))
            raise