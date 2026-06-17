import threading
import time
import uuid
from typing import Any, Dict, List

from agents_driftguard.core.state import AgentState
from agents_driftguard.core.rollback import RollbackResult
from agents_driftguard.core.graph import CausalGraph


class TrackedRun:

    def __init__(self, goal: str) -> None:
        self.run_id = str(uuid.uuid4())
        self.goal = goal
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "running"
        self.states: list[dict[str, Any]] = []
        self.rollbacks: list[dict[str, Any]] = []
        self.graph_mermaid = ""
        self.final_output = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "state_count": len(self.states),
            "rollback_count": len(self.rollbacks),
            "final_output": self.final_output,
        }


class RunTracker:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "RunTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.runs = {}
                cls._instance.active_run_id = None
        return cls._instance

    def start_run(self, goal: str) -> str:
        """Start tracking a new run."""
        run = TrackedRun(goal)
        with self._lock:
            self.runs[run.run_id] = run
            self.active_run_id = run.run_id
        return run.run_id

    def add_state(self, run_id: str, state: AgentState) -> None:
        """Record a state transition."""
        with self._lock:
            if run_id in self.runs:
                # We store a simplified dictionary representation for the dashboard
                s = {
                    "step_id": state.step_id,
                    "timestamp": state.timestamp,
                    "action": state.action,
                    "observation": state.observation,
                    "scratchpad": state.scratchpad,
                    "diverged": state.diverged,
                    "state_hash": state.state_hash,
                    "memory_len": len(state.memory),
                    "inputs": state.inputs,
                    "outputs": state.outputs,
                    "metadata": state.metadata,
                }
                # If state is replacing an existing one (e.g. rollback), we just append for timeline
                self.runs[run_id].states.append(s)

    def mark_diverged(self, run_id: str, step_id: int, reason: str, details: str) -> None:
        """Mark a state as diverged with reason and details."""
        with self._lock:
            if run_id in self.runs:
                run = self.runs[run_id]
                for s in run.states:
                    if s["step_id"] == step_id:
                        s["diverged"] = True
                        s["divergence_reason"] = reason
                        s["divergence_details"] = details
                        break

    def add_rollback(self, run_id: str, rollback: RollbackResult) -> None:
        """Record a rollback event."""
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id].rollbacks.append({
                    "to_step": rollback.rolled_back_to,
                    "steps_undone": rollback.steps_undone,
                    "recovered": rollback.success,
                })

    def update_graph(self, run_id: str, graph: CausalGraph) -> None:
        """Update the latest Mermaid graph."""
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id].graph_mermaid = graph.visualize()

    def finish_run(self, run_id: str, success: bool, final_output: str) -> None:
        """Mark a run as completed."""
        with self._lock:
            if run_id in self.runs:
                run = self.runs[run_id]
                run.end_time = time.time()
                run.status = "success" if success else "failed"
                run.final_output = final_output

    def get_run(self, run_id: str) -> TrackedRun | None:
        with self._lock:
            return self.runs.get(run_id)

    def get_all_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            # Sort by start_time descending
            sorted_runs = sorted(self.runs.values(), key=lambda r: r.start_time, reverse=True)
            return [r.to_dict() for r in sorted_runs]


# Global instance
tracker = RunTracker()
