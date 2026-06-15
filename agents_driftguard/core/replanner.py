"""Replanner with failure-context injection.

After a rollback, the ``Replanner`` generates an injection string that
provides the next prompt with full context about what failed and why,
so the agent can try a different strategy.

The replanner is **LLM-agnostic**: it produces a plain-text string that
the caller injects into the prompt however they wish.
"""

from __future__ import annotations

from agents_driftguard.core.detector import DivergenceSignal
from agents_driftguard.core.state import AgentState


class Replanner:
    """Generates failure-context injection strings for the next agent prompt.

    After a rollback, this class summarises what went wrong and where the
    agent is being restarted from, so the LLM can adjust its strategy.
    """

    # Human-readable reason mapping
    _REASON_EXPLANATIONS: dict[str, str] = {
        "goal_drift": "The agent's actions drifted away from the original goal.",
        "tool_failure": "A tool call returned an error or unexpected result.",
        "loop_detected": "The agent entered a repeated state (infinite loop).",
        "constraint_violation": "A user-defined constraint was violated.",
        "llm_judge": "An LLM judge determined the action was misaligned with the goal.",
    }

    def replan(
        self,
        safe_state: AgentState,
        failed_action: dict[str, object],
        divergence_signal: DivergenceSignal,
        failed_path: list[AgentState],
        original_goal: str,
    ) -> str:
        """Generate a failure-context injection string.

        Args:
            safe_state: The state being rolled back to.
            failed_action: The action dict that caused divergence.
            divergence_signal: The divergence detection result.
            failed_path: All states between ``safe_state`` and the divergence
                point (the "failed path").
            original_goal: The original goal/task description.

        Returns:
            A formatted string ready for prompt injection.
        """
        # Summarise the failed path
        failed_actions_summary = self._summarize_failed_path(failed_path)

        # Human-readable reason
        reason_parts = divergence_signal.reason.split("|")
        reason_explanations = [
            self._REASON_EXPLANATIONS.get(r, r) for r in reason_parts
        ]
        reason_text = " Additionally, ".join(reason_explanations)

        # Build the injection string
        lines = [
            "=== ROLLBACK CONTEXT ===",
            "",
            f"You previously attempted: {self._format_action(failed_action)}",
            f"This failed because: {reason_text}",
            "",
        ]

        if divergence_signal.details:
            lines.append(f"Details: {divergence_signal.details}")
            lines.append("")

        lines.extend([
            f"You have been rolled back to step {safe_state.step_id}.",
            f"Original goal: {original_goal}",
            "",
        ])

        if failed_actions_summary:
            lines.append("Actions that led to failure (do NOT repeat this approach):")
            for action_desc in failed_actions_summary:
                lines.append(f"  - {action_desc}")
            lines.append("")

        lines.extend([
            "Try a different strategy to achieve the goal.",
            "=== END ROLLBACK CONTEXT ===",
        ])

        return "\n".join(lines)

    @staticmethod
    def _format_action(action: dict[str, object]) -> str:
        """Format an action dict into a readable string.

        Args:
            action: The action dict to format.

        Returns:
            Human-readable string describing the action.
        """
        tool = action.get("tool_name", "unknown_tool")
        args = action.get("args", {})
        if args:
            args_str = ", ".join(f"{k}={v!r}" for k, v in (args if isinstance(args, dict) else {}).items())
            return f"{tool}({args_str})"
        return str(tool)

    @staticmethod
    def _summarize_failed_path(failed_path: list[AgentState]) -> list[str]:
        """Summarise each step in the failed path.

        Args:
            failed_path: States that were part of the failed trajectory.

        Returns:
            List of human-readable action descriptions.
        """
        summaries: list[str] = []
        for state in failed_path:
            tool = state.action.get("tool_name", "unknown")
            obs_preview = state.observation[:80] + "..." if len(state.observation) > 80 else state.observation
            summaries.append(
                f"Step {state.step_id}: {tool} → {obs_preview!r}"
            )
        return summaries
