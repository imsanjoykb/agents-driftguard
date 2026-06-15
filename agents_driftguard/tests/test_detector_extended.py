import pytest

from agents_driftguard.core.detector import (
    HallucinationDetector,
    OutputQualityDetector,
    TokenBudgetDetector,
)
from agents_driftguard.core.state import AgentState


def _make_state(step_id: int, observation: str = "", scratchpad: str = "") -> AgentState:
    return AgentState(
        step_id=step_id,
        action={"tool_name": "test"},
        observation=observation,
        memory=[],
        scratchpad=scratchpad,
        goal_embedding=[],
        state_embedding=[],
    )


class TestOutputQualityDetector:
    def test_too_short(self) -> None:
        detector = OutputQualityDetector(min_length=10)
        state = _make_state(1, observation="short")
        signal = detector.detect(state, [])
        assert signal.detected
        assert signal.reason == "output_quality"
        assert "too short" in signal.details

    def test_sufficient_length_passes(self) -> None:
        detector = OutputQualityDetector(min_length=10)
        state = _make_state(1, observation="this is long enough")
        signal = detector.detect(state, [])
        assert not signal.detected

    def test_repetition_ratio_exceeded(self) -> None:
        detector = OutputQualityDetector(max_repetition_ratio=0.5)
        state = _make_state(1, observation="error error error error error")
        signal = detector.detect(state, [])
        assert signal.detected
        assert signal.reason == "output_quality"
        assert "High repetition ratio" in signal.details

    def test_repetition_passes(self) -> None:
        detector = OutputQualityDetector(max_repetition_ratio=0.5)
        state = _make_state(1, observation="this is a completely normal observation text")
        signal = detector.detect(state, [])
        assert not signal.detected


class TestTokenBudgetDetector:
    def test_under_budget(self) -> None:
        detector = TokenBudgetDetector(max_tokens=100)
        s1 = _make_state(1, observation="word " * 10, scratchpad="think " * 10)
        s2 = _make_state(2, observation="word " * 10, scratchpad="think " * 10)
        signal = detector.detect(s2, [s1])
        assert not signal.detected

    def test_exceeds_budget(self) -> None:
        detector = TokenBudgetDetector(max_tokens=10)
        s1 = _make_state(1, observation="word " * 6, scratchpad="think " * 6)
        s2 = _make_state(2, observation="word " * 6, scratchpad="think " * 6)
        signal = detector.detect(s2, [s1])
        assert signal.detected
        assert signal.reason == "token_budget"
        assert "Cumulative tokens" in signal.details


class TestHallucinationDetector:
    def test_contradiction_positive_to_negative(self) -> None:
        detector = HallucinationDetector()
        s1 = _make_state(1, observation="The main database server is disconnected and unreachable")
        s2 = _make_state(2, observation="The main database server is connected and running fine")
        signal = detector.detect(s2, [s1])
        assert signal.detected
        assert signal.reason == "hallucination"
        assert "Contradiction:" in signal.details
        assert "connected" in signal.details
        assert "disconnected" in signal.details

    def test_contradiction_negative_to_positive(self) -> None:
        detector = HallucinationDetector()
        s1 = _make_state(1, observation="User file exists in the directory")
        s2 = _make_state(2, observation="User file does not exist anywhere")
        signal = detector.detect(s2, [s1])
        assert signal.detected
        assert signal.reason == "hallucination"
        assert "Contradiction:" in signal.details

    def test_no_contradiction_passes(self) -> None:
        detector = HallucinationDetector()
        s1 = _make_state(1, observation="Server is connected and running fine")
        s2 = _make_state(2, observation="User file exists in the directory")
        signal = detector.detect(s2, [s1])
        assert not signal.detected

    def test_requires_content_word_overlap(self) -> None:
        detector = HallucinationDetector()
        # "available" vs "unavailable" but about different things
        s1 = _make_state(1, observation="Database cluster is unavailable")
        s2 = _make_state(2, observation="User profile is available")
        signal = detector.detect(s2, [s1])
        assert not signal.detected  # No overlap in content words (database/cluster vs user/profile)
