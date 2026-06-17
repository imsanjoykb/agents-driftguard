from __future__ import annotations
from agents_driftguard.core.detector import DivergenceSignal
from agents_driftguard.core.state import AgentState


class Replanner:

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
        tool = action.get("tool_name", "unknown_tool")
        args = action.get("args", {})
        if args:
            args_str = ", ".join(f"{k}={v!r}" for k, v in (args if isinstance(args, dict) else {}).items())
            return f"{tool}({args_str})"
        return str(tool)

    @staticmethod
    def _summarize_failed_path(failed_path: list[AgentState]) -> list[str]:
        summaries: list[str] = []
        for state in failed_path:
            tool = state.action.get("tool_name", "unknown")
            obs_preview = state.observation[:80] + "..." if len(state.observation) > 80 else state.observation
            summaries.append(
                f"Step {state.step_id}: {tool} → {obs_preview!r}"
            )
        return summaries
