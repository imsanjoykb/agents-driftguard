"""agents_driftguard — Causal rollback and replanning for LLM agents.

Wraps any agent executor with a causal state graph, divergence detection,
automatic rollback, and failure-context-aware replanning.

Quick-start::

    from agents_driftguard import AgentUndoWrapper

    def my_agent(prompt: str) -> str:
        # your LLM call here
        return response

    wrapper = AgentUndoWrapper(executor=my_agent)
    result = wrapper.run(goal="Accomplish some task")
    print(result.final_output)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from agents_driftguard.core.detector import (
    BaseDetector,
    ConstraintChecker,
    DivergenceDetector,
    DivergenceSignal,
    GoalDriftDetector,
    LLMJudge,
    LoopDetector,
    ToolFailureDetector,
    OutputQualityDetector,
    TokenBudgetDetector,
    HallucinationDetector,
)
from agents_driftguard.core.graph import CausalGraph
from agents_driftguard.core.replanner import Replanner
from agents_driftguard.core.rollback import (
    RollbackController,
    RollbackEvent,
    RollbackResult,
    ReversibleToolRegistry,
    get_global_registry,
    reversible,
)
from agents_driftguard.core.state import AgentState, StateStore
from agents_driftguard.store.memory import InMemoryStateStore
from agents_driftguard.utils.embeddings import (
    EmbeddingBackend,
    cosine_similarity,
    get_embedder,
)
from agents_driftguard.utils.hashing import compute_state_hash

logger = logging.getLogger("agents_driftguard")

__version__ = "1.1.0"


# ---------------------------------------------------------------------------
# Agent run result
# ---------------------------------------------------------------------------

class AgentRunResult(BaseModel):
    """Final result of an agent run managed by ``AgentUndoWrapper``.

    Attributes:
        success: Whether the agent completed its goal successfully.
        final_output: The last output produced by the agent.
        total_steps: Total number of steps executed (including re-tries).
        rollbacks_performed: Number of rollbacks triggered during the run.
        causal_graph_mermaid: Mermaid diagram string of the causal graph.
        rollback_history: Details of every rollback that occurred.
    """

    success: bool = False
    final_output: str = ""
    total_steps: int = 0
    rollbacks_performed: int = 0
    causal_graph_mermaid: str = ""
    rollback_history: list[RollbackResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step parser result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Parsed result of a single executor step.

    Attributes:
        action: The action dict (tool_name, args, etc.).
        observation: The tool output / result string.
        scratchpad: Agent reasoning trace.
        done: Whether the agent signals it has finished.
        raw_output: The raw output from the executor.
    """

    action: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    scratchpad: str = ""
    done: bool = False
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Default step parser
# ---------------------------------------------------------------------------

def _default_parse_step(raw_output: str) -> StepResult:
    """Parse a raw executor output string into a ``StepResult``.

    This is a best-effort parser for simple agent outputs. For structured
    frameworks (LangChain, OpenAI), use the respective integration adapter.

    The parser looks for special markers:
    - ``ACTION:`` line → parsed as tool_name
    - ``OBSERVATION:`` line → tool output
    - ``THOUGHT:`` line → scratchpad
    - ``DONE`` or ``FINAL ANSWER:`` → marks completion

    If no markers are found, the entire output is treated as an observation.

    Args:
        raw_output: The raw string from the executor.

    Returns:
        A ``StepResult`` with parsed fields.
    """
    lines = raw_output.strip().split("\n")
    action: dict[str, Any] = {}
    observation = ""
    scratchpad = ""
    done = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("ACTION:"):
            action["tool_name"] = stripped[7:].strip()
        elif upper.startswith("OBSERVATION:"):
            observation = stripped[12:].strip()
        elif upper.startswith("THOUGHT:"):
            scratchpad = stripped[8:].strip()
        elif upper.startswith("FINAL ANSWER:") or upper == "DONE":
            done = True
            if upper.startswith("FINAL ANSWER:"):
                observation = stripped[13:].strip()

    # Fallback: if no markers found, treat entire output as observation
    if not action and not observation and not scratchpad:
        observation = raw_output.strip()
        # Check for common completion signals
        if observation.lower().startswith("final answer:"):
            done = True
            observation = observation[13:].strip()

    return StepResult(
        action=action,
        observation=observation,
        scratchpad=scratchpad,
        done=done,
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class AgentUndoWrapper:
    """Wraps any agent executor with rollback-aware state management.

    The wrapper maintains a causal graph of agent state transitions,
    monitors for divergence after each step, and automatically rolls back
    to the last safe state when problems are detected — then replans with
    failure context injected into the next prompt.

    Args:
        executor: A callable ``(prompt: str) -> str`` that runs one agent step.
        max_rollbacks: Maximum number of rollbacks before giving up.
        detectors: List of detector names to enable (e.g. ``["goal_drift", "tool_failure"]``).
        goal_drift_threshold: Cosine similarity threshold for goal-drift detection.
        store: State store backend — ``"memory"`` or ``"sqlite"`` (or pass an instance).
        constraints: User-defined constraint functions ``fn(state) -> bool``.
        on_rollback: Optional callback invoked on each rollback.
        embedding_backend: Embedding backend name (``"dummy"``, ``"sentence-transformers"``, ``"openai"``).
        step_parser: Custom function to parse raw executor output into ``StepResult``.
        max_steps: Maximum total steps before forced termination.
        verbose: If ``True``, emit debug-level logging.
    """

    def __init__(
        self,
        executor: Callable[[str], str],
        *,
        max_rollbacks: int = 3,
        detectors: list[str] | None = None,
        goal_drift_threshold: float = 0.75,
        store: str | Any = "memory",
        constraints: list[Callable[[AgentState], bool]] | None = None,
        on_rollback: Callable[[RollbackResult], None] | None = None,
        embedding_backend: str = "dummy",
        step_parser: Callable[[str], StepResult] | None = None,
        max_steps: int = 50,
        verbose: bool = False,
    ) -> None:
        self.executor = executor
        self.max_rollbacks = max_rollbacks
        self.goal_drift_threshold = goal_drift_threshold
        self.on_rollback = on_rollback
        self.step_parser = step_parser or _default_parse_step
        self.max_steps = max_steps
        self.verbose = verbose

        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # State store
        if isinstance(store, str):
            self.store = self._create_store(store)
        else:
            self.store = store

        # Embedding backend
        self.embedder: EmbeddingBackend = get_embedder(embedding_backend)
        from agents_driftguard.utils.embeddings import DummyEmbeddingBackend
        if isinstance(self.embedder, DummyEmbeddingBackend):
            self.goal_drift_threshold = -1.0

        # Causal graph
        self.graph = CausalGraph()

        # Rollback controller
        self.rollback_controller = RollbackController(verbose=verbose)

        # Replanner
        self.replanner = Replanner()

        # Build detectors
        detector_names = detectors if detectors is not None else ["goal_drift", "tool_failure", "loop"]
        self._detector = self._build_detector(
            detector_names,
            goal_drift_threshold,
            constraints or [],
        )

    @staticmethod
    def _create_store(store_type: str) -> Any:
        """Create a state store by name.

        Args:
            store_type: ``"memory"`` or ``"sqlite"``.

        Returns:
            A state store instance.
        """
        if store_type == "memory":
            return InMemoryStateStore()
        if store_type == "sqlite":
            from agents_driftguard.store.sqlite import SQLiteStateStore
            return SQLiteStateStore()
        raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory' or 'sqlite'.")

    @staticmethod
    def _build_detector(
        names: list[str],
        goal_drift_threshold: float,
        constraints: list[Callable[[AgentState], bool]],
    ) -> DivergenceDetector:
        """Build the composite divergence detector from names.

        Args:
            names: List of detector names to enable.
            goal_drift_threshold: Threshold for goal-drift.
            constraints: User-defined constraint functions.

        Returns:
            A configured ``DivergenceDetector`` aggregator.
        """
        sub_detectors: list[BaseDetector] = []
        for name in names:
            if name == "goal_drift":
                sub_detectors.append(GoalDriftDetector(absolute_threshold=goal_drift_threshold))
            elif name == "tool_failure":
                sub_detectors.append(ToolFailureDetector())
            elif name == "loop":
                sub_detectors.append(LoopDetector())
            elif name == "constraint":
                sub_detectors.append(ConstraintChecker(constraints=constraints))
            elif name == "llm_judge":
                # Requires passing a custom judge_fn via constraints or separate mechanism,
                # omitting here for simplicity; usually initialized separately.
                pass
            elif name == "output_quality":
                sub_detectors.append(OutputQualityDetector())
            elif name == "token_budget":
                sub_detectors.append(TokenBudgetDetector())
            elif name == "hallucination":
                sub_detectors.append(HallucinationDetector())
            else:
                logger.warning("Unknown detector %r, skipping", name)
        return DivergenceDetector(detectors=sub_detectors)

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self, goal: str) -> AgentRunResult:
        """Execute the agent with rollback-aware state management.

        This is the main entry point. It:

        1. Embeds the goal.
        2. Loops: call executor → parse step → build state → embed →
           add to graph → detect divergence → rollback/replan if needed.
        3. Returns when the agent signals completion, max steps is reached,
           or max rollbacks is exceeded.

        Args:
            goal: The task/goal description for the agent.

        Returns:
            An ``AgentRunResult`` summarising the run.
        """
        logger.info("Starting agent run with goal: %s", goal[:100])

        # Embed the goal once
        goal_embedding = self.embedder.embed(goal)

        # Initialize tracking
        step_counter = 0
        rollback_count = 0
        rollback_history: list[RollbackResult] = []
        current_prompt = goal
        injection_context = ""
        last_output = ""

        # Create initial state (S0)
        initial_state = AgentState(
            step_id=0,
            action={"tool_name": "__start__", "type": "init"},
            observation=goal,
            memory=[{"role": "user", "content": goal}],
            scratchpad="Agent initialized with goal.",
            goal_embedding=goal_embedding,
            state_embedding=self.embedder.embed(goal),
            metadata={"is_initial": True},
        )
        self.store.save(initial_state)
        self.graph.add_state(initial_state)
        
        # Dashboard tracking
        from agents_driftguard.dashboard.tracker import tracker
        run_id = tracker.start_run(goal)
        tracker.add_state(run_id, initial_state)
        tracker.update_graph(run_id, self.graph)
        
        prev_state = initial_state

        while step_counter < self.max_steps:
            step_counter += 1
            logger.debug("Step %d", step_counter)

            # Build prompt with optional injection context
            prompt = current_prompt
            if injection_context:
                prompt = f"{injection_context}\n\n{current_prompt}"
                injection_context = ""  # clear after use

            # 1. Call executor
            try:
                raw_output = self.executor(prompt)
            except Exception as exc:
                raw_output = f"OBSERVATION: Error executing agent: {exc}"
                logger.error("Executor raised: %s", exc)

            # 2. Parse step
            step_result = self.step_parser(raw_output)
            last_output = step_result.observation or step_result.raw_output

            # Auto-extract inputs/outputs dependencies
            inputs = []
            outputs = []
            action_args = step_result.action.get("args", {}) if step_result.action else {}
            tool_name = step_result.action.get("tool_name", "") if step_result.action else ""
            
            # Look for resource arguments
            resource_keys = ["path", "file", "filename", "table", "db", "db_name", "url", "resource"]
            resource_val = None
            if isinstance(action_args, dict):
                for rk in resource_keys:
                    if rk in action_args:
                        resource_val = str(action_args[rk])
                        break
                if not resource_val and "query" in action_args:
                    resource_val = str(action_args["query"])

            if resource_val:
                # Classify as input vs output based on tool name keywords
                is_write = any(w in tool_name.lower() or w in step_result.action.get("type", "").lower() for w in ["write", "export", "save", "update", "delete", "clean"])
                if is_write:
                    outputs.append(resource_val)
                else:
                    inputs.append(resource_val)

            # Heuristically scan observation text for saved/written resource names
            if not outputs and step_result.observation:
                match = re.search(r"(?i)\b(?:saved|written|exported)\s+to\s+([a-zA-Z0-9_\-\.\/]+)\b", step_result.observation)
                if match:
                    outputs.append(match.group(1))

            # 3. Build AgentState
            memory = prev_state.memory.copy()
            memory.append({"role": "assistant", "content": raw_output})

            state_text = f"{step_result.action} {step_result.observation} {step_result.scratchpad}"
            state_embedding = self.embedder.embed(state_text)

            state = AgentState(
                step_id=step_counter,
                action=step_result.action,
                observation=step_result.observation,
                memory=memory,
                scratchpad=step_result.scratchpad,
                goal_embedding=goal_embedding,
                state_embedding=state_embedding,
                inputs=inputs,
                outputs=outputs,
                metadata={"raw_output": raw_output},
            )

            # 4. Save and add to graph
            self.store.save(state)
            self.graph.add_transition(prev_state, state)
            tracker.add_state(run_id, state)
            tracker.update_graph(run_id, self.graph)

            # 5. Check for completion
            if step_result.done:
                logger.info("Agent signalled completion at step %d", step_counter)
                tracker.finish_run(run_id, True, last_output)
                return AgentRunResult(
                    success=True,
                    final_output=last_output,
                    total_steps=step_counter,
                    rollbacks_performed=rollback_count,
                    causal_graph_mermaid=self.graph.visualize(),
                    rollback_history=rollback_history,
                )

            # 6. Run divergence detection
            history = self.store.get_all()
            signal = self._detector.detect(state, history[:-1])  # exclude current

            if signal.detected:
                logger.warning(
                    "Divergence detected at step %d: %s (confidence=%.2f)",
                    step_counter, signal.reason, signal.confidence,
                )

                # Check rollback budget
                if rollback_count >= self.max_rollbacks:
                    logger.error("Max rollbacks (%d) exceeded", self.max_rollbacks)
                    final_output = f"Max rollbacks exceeded. Last output: {last_output}"
                    tracker.finish_run(run_id, False, final_output)
                    return AgentRunResult(
                        success=False,
                        final_output=final_output,
                        total_steps=step_counter,
                        rollbacks_performed=rollback_count,
                        causal_graph_mermaid=self.graph.visualize(),
                        rollback_history=rollback_history,
                    )

                # 6a. Mark diverged
                self.graph.mark_node_diverged(step_counter)
                tracker.mark_diverged(run_id, step_counter, signal.reason, signal.details)

                # 6b. Find safe rollback point
                safe_state = self.graph.get_safe_rollback_point(
                    step_counter,
                    goal_sim_threshold=self.goal_drift_threshold,
                    failed_resource=signal.failed_resource,
                )
                if safe_state is None:
                    # Fallback to initial state
                    safe_state = initial_state
                    logger.warning("No safe rollback point found, falling back to S0")

                # 6c. Rollback
                states_to_undo = [
                    s for s in self.store.get_all()
                    if s.step_id > safe_state.step_id
                ]
                result = self.rollback_controller.rollback(
                    to_state=safe_state,
                    store=self.store,
                    states_to_undo=states_to_undo,
                )
                rollback_count += 1
                rollback_history.append(result)
                tracker.add_rollback(run_id, result)

                if self.on_rollback:
                    self.on_rollback(result)

                logger.info(
                    "Rolled back to step %d (%d steps undone)",
                    safe_state.step_id, result.steps_undone,
                )

                # 6d. Replan
                failed_path = [
                    s for s in states_to_undo if not s.diverged
                ]
                injection_context = self.replanner.replan(
                    safe_state=safe_state,
                    failed_action=state.action,
                    divergence_signal=signal,
                    failed_path=failed_path,
                    original_goal=goal,
                )

                # Reset to safe state
                prev_state = safe_state
                current_prompt = goal
                continue

            # Normal progression
            prev_state = state
            current_prompt = goal  # In a real agent, this would evolve

        # Max steps exceeded
        logger.warning("Max steps (%d) exceeded", self.max_steps)
        final_output = f"Max steps exceeded. Last output: {last_output}"
        tracker.finish_run(run_id, False, final_output)
        return AgentRunResult(
            success=False,
            final_output=final_output,
            total_steps=step_counter,
            rollbacks_performed=rollback_count,
            causal_graph_mermaid=self.graph.visualize(),
            rollback_history=rollback_history,
        )


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # Main API
    "AgentUndoWrapper",
    "AgentRunResult",
    "StepResult",
    # State
    "AgentState",
    "StateStore",
    # Graph
    "CausalGraph",
    # Detection
    "DivergenceDetector",
    "DivergenceSignal",
    "BaseDetector",
    "GoalDriftDetector",
    "ToolFailureDetector",
    "LoopDetector",
    "ConstraintChecker",
    "LLMJudge",
    "OutputQualityDetector",
    "TokenBudgetDetector",
    "HallucinationDetector",
    # Rollback
    "RollbackController",
    "RollbackResult",
    "RollbackEvent",
    "ReversibleToolRegistry",
    "reversible",
    "get_global_registry",
    # Replanner
    "Replanner",
    # Embeddings
    "EmbeddingBackend",
    "cosine_similarity",
    "get_embedder",
    # Stores
    "InMemoryStateStore",
    # Hashing
    "compute_state_hash",
    # Version
    "__version__",
]
