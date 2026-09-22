from __future__ import annotations
import asyncio
import structlog

from src.pipeline.orchestrator import VoiceAgentOrchestrator
from src.vad.silero_vad import SileroVAD
from src.stt.deepgram_stt import DeepgramSTT
from src.llm.groq_llm import GroqLLM
from src.tts.elevenlabs_tts import ElevenLabsTTS

log = structlog.get_logger()

def build_orchestrator() -> VoiceAgentOrchestrator:
    # Factory - assembles a fresh orchestrator with real providers for one session
    # A new instance per connection keeps sessions fully isolated (no shared state, no cross-talk between concurrent users)
    return VoiceAgentOrchestrator(vad=SileroVAD(), stt=DeepgramSTT(), llm=GroqLLM(), tts=ElevenLabsTTS(),)