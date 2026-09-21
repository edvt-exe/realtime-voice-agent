import asyncio
from src.pipeline.orchestrator import VoiceAgentOrchestrator
from src.pipeline.interfaces import TranscriptEvent, VADEvent

class SlowLLM:
    async def stream(self, messages):
        for tok in ["This ", "is ", "a ", "slow ", "response", "."]:
            await asyncio.sleep(0.05)
            yield tok

class MockTTS:
    async def stream(self, text):
        yield b"\x00\x01"

class MockSTT:
    async def stream(self, audio_in):
        yield TranscriptEvent(text="hello", is_final=True)

class BargeInVAD:
    async def stream(self, audio_in):
        await asyncio.sleep(0.1)
        yield VADEvent(type="speech_start", timestamp=0.0)

async def main():
    orch = VoiceAgentOrchestrator(vad=BargeInVAD(), stt=MockSTT(), llm=SlowLLM(), tts=MockTTS())
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.3)
    task.cancel()
    print("Conversation history after barge-in:", orch.conversation_history)

asyncio.run(main())