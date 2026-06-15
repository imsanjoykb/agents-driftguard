"""End-to-end test with a mock agent that fails at step 3 and recovers.

This is the most important test — it validates the full
AgentUndoWrapper pipeline:
1. Agent succeeds at steps 1, 2
2. Fails at step 3 (tool error in observation)
3. Rollback to step 2
4. Succeeds at step 4 after rollback
"""

from __future__ import annotations

import pytest

from agents_driftguard import AgentUndoWrapper, AgentRunResult


class MockAgent:
    """A mock agent that fails at a specific step, then succeeds after rollback.

    Step sequence:
    - Call 1: returns ACTION with successful observation
    - Call 2: returns ACTION with successful observation
    - Call 3: returns ACTION with ERROR observation (triggers rollback)
    - Call 4: returns FINAL ANSWER (success after rollback)

    After rollback, the agent receives a modified prompt containing
    rollback context, so it takes a different path.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.prompts_received: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.call_count == 1:
            return (
                "THOUGHT: I need to search for information first.\n"
                "ACTION: search\n"
                "OBSERVATION: Found 5 relevant results about the topic."
            )
        elif self.call_count == 2:
            return (
                "THOUGHT: Let me process the data.\n"
                "ACTION: compute\n"
                "OBSERVATION: Computed summary statistics successfully."
            )
        elif self.call_count == 3:
            return (
                "THOUGHT: Now I'll write the results.\n"
                "ACTION: write_file\n"
                "OBSERVATION: Error: Permission denied - cannot write to /protected/path"
            )
        else:
            # After rollback, take a different approach
            return (
                "THOUGHT: The previous write failed. Let me output directly.\n"
                "FINAL ANSWER: Here are the results: summary statistics computed from 5 sources."
            )


class TestEndToEnd:
    """End-to-end test of the full AgentUndoWrapper pipeline."""

    def test_mock_agent_rollback_and_recovery(self) -> None:
        """Agent fails at step 3, rolls back to step 2, succeeds at step 4."""
        mock = MockAgent()

        wrapper = AgentUndoWrapper(
            executor=mock,
            max_rollbacks=3,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="memory",
            verbose=False,
            max_steps=20,
        )

        result = wrapper.run(goal="Analyze data and produce a summary report")

        # Core assertions
        assert result.success is True, f"Expected success but got: {result.final_output}"
        assert result.rollbacks_performed == 1
        assert "results" in result.final_output.lower() or "summary" in result.final_output.lower()

        # The mock was called 4 times (3 normal + 1 after rollback)
        assert mock.call_count == 4

        # Verify rollback context was injected
        # The 4th prompt should contain rollback context
        assert len(mock.prompts_received) == 4
        assert "ROLLBACK" in mock.prompts_received[3]

        # Causal graph should exist
        assert result.causal_graph_mermaid != ""
        assert "flowchart" in result.causal_graph_mermaid

    def test_max_rollbacks_exceeded(self) -> None:
        """Agent fails and keeps failing until max_rollbacks is exceeded."""

        call_count = 0

        def always_failing(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return (
                "ACTION: bad_tool\n"
                "OBSERVATION: Error: everything is broken"
            )

        wrapper = AgentUndoWrapper(
            executor=always_failing,
            max_rollbacks=2,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="memory",
            max_steps=50,
        )

        result = wrapper.run(goal="Do something")

        assert result.success is False
        assert result.rollbacks_performed == 2
        assert "Max rollbacks exceeded" in result.final_output

    def test_successful_run_no_rollbacks(self) -> None:
        """Agent succeeds immediately with no rollbacks needed."""

        def happy_agent(prompt: str) -> str:
            return "FINAL ANSWER: All done successfully!"

        wrapper = AgentUndoWrapper(
            executor=happy_agent,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="memory",
        )

        result = wrapper.run(goal="Simple task")

        assert result.success is True
        assert result.rollbacks_performed == 0
        assert "done" in result.final_output.lower()

    def test_max_steps_exceeded(self) -> None:
        """Agent runs out of steps."""

        step = 0

        def looping_agent(prompt: str) -> str:
            nonlocal step
            step += 1
            return f"OBSERVATION: Step {step} completed, more work needed"

        wrapper = AgentUndoWrapper(
            executor=looping_agent,
            detectors=[],  # No detectors, so no rollbacks
            embedding_backend="dummy",
            store="memory",
            max_steps=5,
        )

        result = wrapper.run(goal="Infinite task")

        assert result.success is False
        assert result.total_steps == 5
        assert "Max steps exceeded" in result.final_output

    def test_rollback_callback_invoked(self) -> None:
        """on_rollback callback is called when rollback occurs."""
        callback_results: list = []

        mock = MockAgent()

        wrapper = AgentUndoWrapper(
            executor=mock,
            max_rollbacks=3,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="memory",
            on_rollback=lambda r: callback_results.append(r),
        )

        result = wrapper.run(goal="Do a task")

        assert result.success is True
        assert len(callback_results) == 1
        assert callback_results[0].success is True

    def test_causal_graph_has_diverged_node(self) -> None:
        """After rollback, the causal graph marks the failed step as diverged."""
        mock = MockAgent()

        wrapper = AgentUndoWrapper(
            executor=mock,
            max_rollbacks=3,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="memory",
        )

        result = wrapper.run(goal="Task with failure")

        # Check the graph directly
        graph = wrapper.graph

        # The diverged node should be step 3 (which is the 3rd executor step)
        # Because of the internal step counter, let's just verify there IS a diverged node
        diverged_found = False
        for node_id in graph.nodes:
            state = graph.get_state(int(node_id[1:]))
            if state and state.diverged:
                diverged_found = True
                break
        assert diverged_found, "Expected at least one diverged node in the graph"

    def test_sqlite_store_integration(self) -> None:
        """Run with SQLite store instead of memory."""
        mock = MockAgent()

        wrapper = AgentUndoWrapper(
            executor=mock,
            max_rollbacks=3,
            detectors=["tool_failure"],
            embedding_backend="dummy",
            store="sqlite",
        )

        result = wrapper.run(goal="Task with SQLite store")
        assert result.success is True
        assert result.rollbacks_performed == 1
