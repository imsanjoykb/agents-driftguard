from __future__ import annotations
import pytest
from agents_driftguard.core.detector import (
    ConstraintChecker,
    DivergenceDetector,
    GoalDriftDetector,
    LLMJudge,
    LoopDetector,
    ToolFailureDetector,
)
from agents_driftguard.core.state import AgentState

def _make_state(
    step_id: int,
    observation: str = "ok",
    goal_emb: list[float] | None = None,
    state_emb: list[float] | None = None,
    state_hash: str = "",
) -> AgentState:
    """Helper to create a test state for detector tests."""
    return AgentState(
        step_id=step_id,
        action={"tool_name": "test_tool"},
        observation=observation,
        goal_embedding=goal_emb or [],
        state_embedding=state_emb or [],
        state_hash=state_hash or f"hash_{step_id}",
    )


class TestGoalDriftDetector:
    """Test the GoalDriftDetector."""

    def test_no_drift_when_aligned(self) -> None:
        """No divergence when embeddings are similar."""
        det = GoalDriftDetector(absolute_threshold=0.75)
        state = _make_state(1, goal_emb=[1.0, 0.0, 0.0], state_emb=[0.9, 0.1, 0.0])
        signal = det.detect(state, [])
        assert not signal.detected

    def test_drift_below_threshold(self) -> None:
        """Fires when similarity drops below absolute threshold."""
        det = GoalDriftDetector(absolute_threshold=0.75)
        # Orthogonal vectors → cosine sim ≈ 0
        state = _make_state(1, goal_emb=[1.0, 0.0, 0.0], state_emb=[0.0, 1.0, 0.0])
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.reason == "goal_drift"

    def test_large_step_drop(self) -> None:
        """Fires when similarity drops sharply in one step."""
        det = GoalDriftDetector(absolute_threshold=0.5, step_drop_threshold=0.2)
        prev = _make_state(0, goal_emb=[1.0, 0.0], state_emb=[0.95, 0.05])
        curr = _make_state(1, goal_emb=[1.0, 0.0], state_emb=[0.3, 0.7])
        signal = det.detect(curr, [prev])
        assert signal.detected
        assert signal.reason == "goal_drift"

    def test_no_embeddings_no_drift(self) -> None:
        """Returns no signal when embeddings are missing."""
        det = GoalDriftDetector()
        state = _make_state(1)
        signal = det.detect(state, [])
        assert not signal.detected


class TestToolFailureDetector:
    """Test the ToolFailureDetector."""

    def test_no_error(self) -> None:
        """Normal observation passes."""
        det = ToolFailureDetector()
        state = _make_state(1, observation="5 results found")
        signal = det.detect(state, [])
        assert not signal.detected

    def test_empty_observation(self) -> None:
        """Empty observation triggers detection."""
        det = ToolFailureDetector()
        state = _make_state(1, observation="")
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.reason == "tool_failure"

    def test_error_keyword(self) -> None:
        """Error keyword in observation triggers detection."""
        det = ToolFailureDetector()
        state = _make_state(1, observation="ConnectionError: failed to connect")
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.reason == "tool_failure"

    def test_http_error_code(self) -> None:
        """HTTP error code triggers detection."""
        det = ToolFailureDetector()
        state = _make_state(1, observation="HTTP 500 Internal Server Error")
        signal = det.detect(state, [])
        assert signal.detected

    def test_traceback(self) -> None:
        """Python traceback triggers detection."""
        det = ToolFailureDetector()
        state = _make_state(1, observation="Traceback (most recent call last):")
        signal = det.detect(state, [])
        assert signal.detected


class TestLoopDetector:
    """Test the LoopDetector."""

    def test_no_loop(self) -> None:
        """Unique hashes pass."""
        det = LoopDetector(window=3)
        history = [_make_state(i, state_hash=f"hash_{i}") for i in range(3)]
        current = _make_state(3, state_hash="hash_3")
        signal = det.detect(current, history)
        assert not signal.detected

    def test_loop_detected(self) -> None:
        """Repeated hash within window triggers detection."""
        det = LoopDetector(window=3)
        history = [_make_state(i, state_hash=f"hash_{i}") for i in range(3)]
        current = _make_state(3, state_hash="hash_1")  # same as step 1
        signal = det.detect(current, history)
        assert signal.detected
        assert signal.reason == "loop_detected"

    def test_empty_history(self) -> None:
        """No loop with empty history."""
        det = LoopDetector()
        state = _make_state(0)
        signal = det.detect(state, [])
        assert not signal.detected


class TestConstraintChecker:
    """Test the ConstraintChecker."""

    def test_all_constraints_pass(self) -> None:
        """No divergence when all constraints pass."""
        det = ConstraintChecker(constraints=[
            lambda s: s.step_id >= 0,
            lambda s: len(s.observation) > 0,
        ])
        state = _make_state(1, observation="ok")
        signal = det.detect(state, [])
        assert not signal.detected

    def test_constraint_violation(self) -> None:
        """Fires when a constraint returns False."""
        det = ConstraintChecker(constraints=[
            lambda s: s.step_id < 5,  # pass
            lambda s: "forbidden" not in s.observation,  # fail
        ])
        state = _make_state(1, observation="forbidden content")
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.reason == "constraint_violation"

    def test_constraint_exception(self) -> None:
        """Fires when a constraint raises an exception."""
        def bad_constraint(s: AgentState) -> bool:
            raise ValueError("boom")

        det = ConstraintChecker(constraints=[bad_constraint])
        state = _make_state(1)
        signal = det.detect(state, [])
        assert signal.detected
        assert "exception" in signal.details.lower()

    def test_no_constraints(self) -> None:
        """No divergence with empty constraint list."""
        det = ConstraintChecker(constraints=[])
        state = _make_state(1)
        signal = det.detect(state, [])
        assert not signal.detected


class TestLLMJudge:
    def test_high_score_passes(self) -> None:
        """No divergence when judge scores high."""
        det = LLMJudge(judge_fn=lambda prompt: 8.0, threshold=5.0)
        state = _make_state(1)
        signal = det.detect(state, [])
        assert not signal.detected

    def test_low_score_fires(self) -> None:
        """Fires when judge scores below threshold."""
        det = LLMJudge(judge_fn=lambda prompt: 2.0, threshold=5.0)
        state = _make_state(1)
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.reason == "llm_judge"

    def test_no_judge_fn(self) -> None:
        det = LLMJudge()
        state = _make_state(1)
        signal = det.detect(state, [])
        assert not signal.detected

    def test_judge_fn_exception(self) -> None:
        def failing_judge(prompt: str) -> float:
            raise RuntimeError("API down")

        det = LLMJudge(judge_fn=failing_judge)
        state = _make_state(1)
        signal = det.detect(state, [])
        assert signal.detected
        assert signal.confidence == 0.5


class TestDivergenceDetectorAggregator:

    def test_any_strategy(self) -> None:
        det = DivergenceDetector(
            detectors=[
                ToolFailureDetector(),
                LoopDetector(),
            ],
            strategy="any",
        )
        # ToolFailure will fire, LoopDetector won't
        state = _make_state(1, observation="Error: something broke")
        signal = det.detect(state, [])
        assert signal.detected

    def test_all_strategy_not_all_fire(self) -> None:
        det = DivergenceDetector(
            detectors=[
                ToolFailureDetector(),
                LoopDetector(),
            ],
            strategy="all",
        )
        state = _make_state(1, observation="Error: broke")
        signal = det.detect(state, [])
        assert not signal.detected  # Loop didn't fire

    def test_no_detectors(self) -> None:
        det = DivergenceDetector(detectors=[], strategy="any")
        state = _make_state(1)
        signal = det.detect(state, [])
        assert not signal.detected

    def test_merged_signal_has_details(self) -> None:
        det = DivergenceDetector(
            detectors=[
                ToolFailureDetector(),
                ConstraintChecker(constraints=[lambda s: False]),
            ],
            strategy="any",
        )
        state = _make_state(1, observation="Error occurred")
        signal = det.detect(state, [])
        assert signal.detected
        assert "Error" in signal.details or "Constraint" in signal.details
