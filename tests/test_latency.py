import time
from src.pipeline.latency import LatencyMarks

def test_latency_report():
    marks = LatencyMarks(speech_end_ts=time.monotonic())
    time.sleep(0.05)
    marks.stt_final_ts = time.monotonic()
    time.sleep(0.02)
    marks.llm_first_token_ts = time.monotonic()
    time.sleep(0.03)
    marks.tts_first_byte_ts = time.monotonic()

    report = marks.report()
    assert report["vad_to_stt_final_ms"] > 40
    assert report["total_e2e_ms"] > 90
    print(report)

test_latency_report()