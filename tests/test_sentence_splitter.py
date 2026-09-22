import pytest
from src.pipeline.sentence_splitter import IncrementalSentenceSplitter

def feed_all(splitter: IncrementalSentenceSplitter, tokens: list[str]) -> list[str]:
    # Helper: feeds a list of tokens sequentially, collects all emitted sentences
    out: list[str] = []
    for token in tokens:
        out.extend(splitter.feed(token))
    return out


class TestSentenceSplitter:
    def test_single_complete_sentence(self):
        splitter = IncrementalSentenceSplitter()
        out = feed_all(splitter, ["Hello", " world", "."])
        assert out == ["Hello world."]

    def test_multiple_sentences_across_tokens(self):
        splitter = IncrementalSentenceSplitter()
        out = feed_all(splitter, ["Hi", "! ", "How ", "are ", "you", "? ", "Fine", "."])
        assert out == ["Hi!", "How are you?"]
        assert splitter.flush() == "Fine."

    def test_no_output_on_incomplete_sentence(self):
        splitter = IncrementalSentenceSplitter()
        out = feed_all(splitter, ["This ", "is ", "incomplete"])
        assert out == []

    def test_flush_returns_remaining_buffer(self):
        splitter = IncrementalSentenceSplitter()
        feed_all(splitter, ["Trailing ", "text ", "no ", "punctuation"])
        assert splitter.flush() == "Trailing text no punctuation"

    def test_flush_returns_none_when_buffer_empty(self):
        splitter = IncrementalSentenceSplitter()
        feed_all(splitter, ["Complete sentence."])
        splitter.flush()
        assert splitter.flush() is None

    def test_clause_level_fallback_on_long_sentence(self):
        splitter = IncrementalSentenceSplitter(min_chars_for_clause=10)
        out = feed_all(splitter, ["This is a long clause, ", "and it continues"])
        assert out == ["This is a long clause,"]
        assert splitter.flush() == "and it continues"

    def test_clause_fallback_not_triggered_below_min_chars(self):
        splitter = IncrementalSentenceSplitter(min_chars_for_clause=100)
        out = feed_all(splitter, ["Short, ", "clause"])
        assert out == []
        assert splitter.flush() == "Short, clause"

    def test_preserves_punctuation(self):
        splitter = IncrementalSentenceSplitter()
        out = feed_all(splitter, ["Wait", "! ", "Really", "? ", "Yes", "."])
        assert out == ["Wait!", "Really?"]
        assert splitter.flush() == "Yes."

    def test_empty_token_stream(self):
        splitter = IncrementalSentenceSplitter()
        out = feed_all(splitter, [])
        assert out == []
        assert splitter.flush() is None

    def test_reset_after_flush_allows_reuse(self):
        splitter = IncrementalSentenceSplitter()
        feed_all(splitter, ["First sentence."])
        splitter.flush()
        out = feed_all(splitter, ["Second sentence."])
        assert out == ["Second sentence."]