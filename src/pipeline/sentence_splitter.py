import re

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_END = re.compile(r'(?<=[,;:])\s+')


class IncrementalSentenceSplitter:
    # Buffers streamed LLM tokens, yields complete sentences/ clauses ASAP for TTS

    def __init__(self, min_chars_for_clause: int = 20) -> None:
        self._buf: str = ""
        self._min_chars = min_chars_for_clause

    def feed(self, token: str) -> list[str]:
        # call once per token received from the LLM stream and returns a list of complete chunks ready to send to TTS (usually 0 or 1)
        self._buf += token
        out: list[str] = []

        while True:
            match = _SENTENCE_END.search(self._buf)
            if match:
                out.append(self._buf[:match.start() + 1].strip())
                self._buf = self._buf[match.end():]
                continue
            # fall back to clause level split if buffer is large enough
            if len(self._buf) >= self._min_chars:
                cmatch = _CLAUSE_END.search(self._buf)
                if cmatch:
                    out.append(self._buf[:cmatch.start() + 1].strip())
                    self._buf = self._buf[cmatch.end():]
                    continue
            break
        return [s for s in out if s]

    def flush(self) -> str | None:
        # call after the LLM stream ends, to get any trailing partial text
        remainder = self._buf.strip()
        self._buf = ""
        return remainder or None