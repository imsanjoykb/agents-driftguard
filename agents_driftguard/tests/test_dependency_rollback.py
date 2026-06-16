"""Tests for Dependency-Aware Semantic Rollbacks."""

from __future__ import annotations

import pytest
from typing import Any

from agents_driftguard import AgentUndoWrapper, AgentRunResult
from agents_driftguard.core.detector import ToolFailureDetector, DivergenceSignal
from agents_driftguard.core.graph import CausalGraph
from agents_driftguard.core.state import AgentState


def test_detector_failed_resource_extraction() -> None:
    """Verify ToolFailureDetector extracts files and database tables from observation errors."""
    detector = ToolFailureDetector()

    # Create dummy states
    dummy_state = AgentState(
        step_id=1,
        action={"tool_name": "run"},
        observation="",
        memory=[],
    )

    # Scenario A: File permission error with filename in observation
    dummy_state.observation = "Error: Permission denied on resource report.csv. Cannot read file."
    signal = detector.detect(dummy_state, [])
    assert signal.detected is True
    assert signal.failed_resource == "report.csv"

    # Scenario B: Database relation error with table name in observation (containing Error keyword)
    dummy_state.observation = "sqlite3.OperationalError: Error - no such table: customers"
    signal = detector.detect(dummy_state, [])
    assert signal.detected is True
    assert signal.failed_resource == "customers"

    # Scenario C: No obvious resource mentioned
    dummy_state.observation = "Error: unexpected tool crash!"
    signal = detector.detect(dummy_state, [])
    assert signal.detected is True
    assert signal.failed_resource is None


def test_graph_dependency_aware_rollback_points() -> None:
    """Verify CausalGraph finds the correct rollback targets via dependency inputs/outputs."""
    g = CausalGraph()

    # Step 0: start
    s0 = AgentState(
        step_id=0,
        action={"tool_name": "__start__", "type": "init"},
        observation="start",
        goal_embedding=[1.0, 0.0],
        state_embedding=[1.0, 0.0],
    )
    # Step 1: Query API (outputs config.json)
    s1 = AgentState(
        step_id=1,
        action={"tool_name": "query_api", "args": {"path": "config.json"}},
        observation="api results",
        goal_embedding=[1.0, 0.0],
        state_embedding=[1.0, 0.0],
        outputs=["config.json"],
    )
    # Step 2: Unrelated operation
    s2 = AgentState(
        step_id=2,
        action={"tool_name": "compute_stats"},
        observation="stats summary",
        goal_embedding=[1.0, 0.0],
        state_embedding=[1.0, 0.0],
    )
    # Step 3: Parse config.json (inputs config.json)
    s3 = AgentState(
        step_id=3,
        action={"tool_name": "read_json", "args": {"file": "config.json"}},
        observation="Error: config.json is corrupt",
        goal_embedding=[1.0, 0.0],
        state_embedding=[1.0, 0.0],
        inputs=["config.json"],
        diverged=True,
    )

    g.add_transition(s0, s1)
    g.add_transition(s1, s2)
    g.add_transition(s2, s3)
    g.mark_node_diverged(3)

    # 1. Dependency-aware lookup: failed resource 'config.json' was output by S1
    # We should rollback to the step before S1 (which is step 0)
    target = g.get_safe_rollback_point(3, failed_resource="config.json")
    assert target is not None
    assert target.step_id == 0

    # 2. Linear default rollback fallback (if failed_resource is not mentioned/unrelated)
    # Standard rollback logic goes to S2 (most recent safe ancestor)
    target_linear = g.get_safe_rollback_point(3, failed_resource=None)
    assert target_linear is not None
    assert target_linear.step_id == 2


class MockDependencyAgent:
    """Mock agent representing a dependency failure scenario.

    Step 1: Init / Search
    Step 2: write_csv (generates output data.csv)
    Step 3: unrelated_compute (unrelated action)
    Step 4: read_csv (fails with permission denied on data.csv)
    Step 5: recovery path (succeeds)
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.prompts_received: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.call_count == 1:
            return (
                "THOUGHT: Let's search for input parameters.\n"
                "ACTION: search\n"
                "OBSERVATION: API return keys: [val1, val2]"
            )
        elif self.call_count == 2:
            return (
                "THOUGHT: Writing data to CSV.\n"
                "ACTION: write_file\n"
                "ARGS: {'path': 'data.csv'}\n"
                "OBSERVATION: saved to data.csv successfully"
            )
        elif self.call_count == 3:
            return (
                "THOUGHT: Run unrelated calculations.\n"
                "ACTION: compute_average\n"
                "OBSERVATION: Calculated average: 42.5"
            )
        elif self.call_count == 4:
            return (
                "THOUGHT: Let's read the CSV file to verify it.\n"
                "ACTION: read_file\n"
                "ARGS: {'path': 'data.csv'}\n"
                "OBSERVATION: Error: Permission denied reading data.csv"
            )
        else:
            return (
                "THOUGHT: Writing to data.csv failed. I will write to a local fallback instead.\n"
                "FINAL ANSWER: Data processing complete."
            )


def test_dependency_aware_rollback_e2e() -> None:
    """E2E verification of wrapper detecting file dependency and rolling back past unrelated steps."""
    mock = MockDependencyAgent()

    wrapper = AgentUndoWrapper(
        executor=mock,
        max_rollbacks=3,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
    )

    result = wrapper.run(goal="Process and verify user data")

    # Core assertion: result succeeded
    assert result.success is True
    # The rollback should target the state before Step 2 (where data.csv was written)
    # Since failure resource is 'data.csv', rollback should target state S1 (step_id = 1), undoing S2 & S3.
    # Therefore, steps_undone should be 2 (S2 and S3), and rolled_back_to should be 1.
    assert result.rollbacks_performed == 1
    assert len(result.rollback_history) == 1
    assert result.rollback_history[0].rolled_back_to == 1
    assert result.rollback_history[0].steps_undone == 3
    
    assert len(mock.prompts_received) > 4
    # Prompt sent after rollback must contain warning context
    last_prompt = mock.prompts_received[-1]
    assert "ROLLBACK" in last_prompt
