import asyncio
import pytest

from src.pipeline.orchestrator import VoiceAgentOrchestrator
from src.pipeline.interfaces import VADEvent, TranscriptEvent
from tests.conftest import MockVAD, MockSTT, MockLLM, MockTTS

@pytest.mark.asyncio
async def test_full_turn_produces_audio_and_updates_history(mock_tts):
    vad = MockVAD(events=[(0.5, VADEvent(type="speech_end", timestamp=0.0))])
    stt = MockSTT(transcripts=[TranscriptEvent(text="hello there", is_final=True)])
    llm = MockLLM(tokens=["Hi", "! ", "How ", "can ", "I ", "help", "?"])

    orch = VoiceAgentOrchestrator(vad=vad, stt=stt, llm=llm, tts=mock_tts)
    task = asyncio.create_task(orch.run())

    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert orch.conversation_history[0] == {"role": "user", "content": "hello there"}
    assert orch.conversation_history[1]["role"] == "assistant"
    assert "Hi!" in mock_tts.synthesized_texts
    assert not orch.audio_out_q.empty()


@pytest.mark.asyncio
async def test_barge_in_cancels_generation_and_clears_queue(mock_tts):
    vad = MockVAD(events=[(0.05, VADEvent(type="speech_end", timestamp=0.0)), (0.1, VADEvent(type="speech_start", timestamp=0.0))])
    stt = MockSTT(transcripts=[TranscriptEvent(text="tell me a story", is_final=True)])
    llm = MockLLM(tokens=["This ", "is ", "a ", "very ", "long ", "response", "."], delay=0.05)

    orch = VoiceAgentOrchestrator(vad=vad, stt=stt, llm=llm, tts=mock_tts)
    task = asyncio.create_task(orch.run())

    await asyncio.sleep(0.3)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # Generation was interrupted, assistant turn should NOT be finalized into history
    assistant_turns = [m for m in orch.conversation_history if m["role"] == "assistant"]
    assert len(assistant_turns) == 0
    assert orch.audio_out_q.empty()


@pytest.mark.asyncio
async def test_latency_marks_are_recorded(mock_tts):
    vad = MockVAD(events=[(0.05, VADEvent(type="speech_end", timestamp=0.0))])
    stt = MockSTT(transcripts=[TranscriptEvent(text="hi", is_final=True)])
    llm = MockLLM(tokens=["Hello", "."])

    orch = VoiceAgentOrchestrator(vad=vad, stt=stt, llm=llm, tts=mock_tts)
    task = asyncio.create_task(orch.run())

    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    report = orch.latency.report()
    assert "total_e2e_ms" in report
    assert report["total_e2e_ms"] >= 0