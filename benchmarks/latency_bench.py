from __future__ import annotations
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import structlog

from src.pipeline.orchestrator import VoiceAgentOrchestrator
from src.pipeline.interfaces import VADEvent, TranscriptEvent
from src.vad.silero_vad import SileroVAD
from src.stt.deepgram_stt import DeepgramSTT
from src.llm.groq_llm import GroqLLM
from src.tts.elevenlabs_tts import ElevenLabsTTS

log = structlog.get_logger()

_RESULTS_DIR = Path("benchmarks/results")
_TEST_PROMPTS = [
    "What's the weather like today?",
    "Tell me a fun fact about space.",
    "How do I make a good cup of coffee?",
    "What's the capital of Japan?",
    "Give me a quick tip for staying productive.",
]

@dataclass
class BenchmarkRun:
    prompt: str
    vad_to_stt_final_ms: float
    stt_to_llm_ttft_ms: float
    llm_to_tts_first_byte_ms: float
    total_e2e_ms: float

class DirectTranscriptSTT:
    # Bypasses real audio capture for benchmarking: emits a known transcriptimmediately, isolating LLM+TTS latency from mic/VAD/STT variability

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    async def stream(self, audio_in):
        yield TranscriptEvent(text=self._transcript, is_final=True)


class ImmediateVAD:
    # Fires speech end immediately — used to start the latency clock at a known, controlled point for benchmarking

    async def stream(self, audio_in):
        yield VADEvent(type="speech_end", timestamp=time.monotonic())


async def run_single_benchmark(prompt: str) -> BenchmarkRun | None:
    orch = VoiceAgentOrchestrator(
        vad=ImmediateVAD(),
        stt=DirectTranscriptSTT(prompt),
        llm=GroqLLM(),
        tts=ElevenLabsTTS(),
    )

    task = asyncio.create_task(orch.run())
    await asyncio.sleep(5.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    report = orch.latency.report()
    if "total_e2e_ms" not in report:
        log.warning("benchmark_incomplete", prompt=prompt, report=report)
        return None

    return BenchmarkRun(
        prompt=prompt,
        vad_to_stt_final_ms=report.get("vad_to_stt_final_ms", 0.0),
        stt_to_llm_ttft_ms=report.get("stt_to_llm_ttft_ms", 0.0),
        llm_to_tts_first_byte_ms=report.get("llm_to_tts_first_byte_ms", 0.0),
        total_e2e_ms=report["total_e2e_ms"],
    )


def summarize(runs: list[BenchmarkRun]) -> dict:
    e2e_values = [r.total_e2e_ms for r in runs]
    return {
        "n_runs": len(runs),
        "mean_e2e_ms": round(statistics.mean(e2e_values), 1),
        "median_e2e_ms": round(statistics.median(e2e_values), 1),
        "p95_e2e_ms": round(statistics.quantiles(e2e_values, n=20)[18], 1) if len(e2e_values) >= 5 else max(e2e_values),
        "min_e2e_ms": round(min(e2e_values), 1),
        "max_e2e_ms": round(max(e2e_values), 1),
        "mean_ttft_ms": round(statistics.mean(r.stt_to_llm_ttft_ms for r in runs), 1),
        "mean_tts_first_byte_ms": round(statistics.mean(r.llm_to_tts_first_byte_ms for r in runs), 1),
    }


async def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[BenchmarkRun] = []

    for prompt in _TEST_PROMPTS:
        log.info("running_benchmark", prompt=prompt)
        result = await run_single_benchmark(prompt)
        if result:
            runs.append(result)
            log.info("benchmark_result", **asdict(result))

    if not runs:
        log.error("no_successful_runs")
        return

    summary = summarize(runs)
    print("\n=== Latency Benchmark Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output = {
        "summary": summary,
        "runs": [asdict(r) for r in runs],
    }
    output_path = _RESULTS_DIR / f"bench_{int(time.time())}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())