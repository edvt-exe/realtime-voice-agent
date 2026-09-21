import asyncio
from src.pipeline.orchestrator import VoiceAgentOrchestrator
from src.pipeline.interfaces import TranscriptEvent


class MockSTT:
    async def stream(self, audio_in):
        yield TranscriptEvent(text="hello", is_final=True)

class MockLLM:
    async def stream(self, messages):
        for tok in ["Hi", " there", "."]:
            yield tok

class MockTTS:
    async def stream(self, text):
        yield b"\x00\x01"

class MockVAD:
    async def stream(self, audio_in):
        return
        yield

async def main():
    orch = VoiceAgentOrchestrator(vad=MockVAD(), stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.1)
    task.cancel()
    chunk = await orch.audio_out_q.get()
    print("Got audio chunk:", chunk)

asyncio.run(main())