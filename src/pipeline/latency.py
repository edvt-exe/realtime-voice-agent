from __future__ import annotations
import time
from dataclasses import dataclass, field

@dataclass
class LatencyMarks:
    # Timestamps captured at each pipeline stage boundary for one conversational turn
    speech_end_ts: float = 0.0
    stt_final_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    tts_first_byte_ts: float = 0.0

    def report(self) -> dict[str, float]:
        """Returns stage-to-stage deltas in milliseconds. Only includes stages that fired."""
        deltas: dict[str, float] = {}

        if self.speech_end_ts and self.stt_final_ts:
            deltas["vad_to_stt_final_ms"] = (self.stt_final_ts - self.speech_end_ts) * 1000

        if self.stt_final_ts and self.llm_first_token_ts:
            deltas["stt_to_llm_ttft_ms"] = (self.llm_first_token_ts - self.stt_final_ts) * 1000

        if self.llm_first_token_ts and self.tts_first_byte_ts:
            deltas["llm_to_tts_first_byte_ms"] = (self.tts_first_byte_ts - self.llm_first_token_ts) * 1000

        if self.speech_end_ts and self.tts_first_byte_ts:
            deltas["total_e2e_ms"] = (self.tts_first_byte_ts - self.speech_end_ts) * 1000

        return deltas

    def reset(self) -> None:
        self.speech_end_ts = 0.0
        self.stt_final_ts = 0.0
        self.llm_first_token_ts = 0.0
        self.tts_first_byte_ts = 0.0