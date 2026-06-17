"""Asynchronous support for agents_driftguard."""

import logging
from typing import Any, Awaitable, Callable

from agents_driftguard.core.detector import DivergenceDetector
from agents_driftguard.core.graph import CausalGraph
from agents_driftguard.core.replanner import Replanner
from agents_driftguard.core.rollback import RollbackController, RollbackResult
from agents_driftguard.core.state import AgentState, StateStore
from agents_driftguard.store.memory import InMemoryStateStore
from agents_driftguard.utils.embeddings import EmbeddingBackend, get_embedder
from agents_driftguard.__init__ import (
    AgentRunResult,
    AgentUndoWrapper,
    StepResult,
    _default_parse_step,
)

logger = logging.getLogger("agents_driftguard.async")


class AsyncAgentUndoWrapper(AgentUndoWrapper):

    def __init__(
        self,
        executor: Callable[[str], Awaitable[str]],
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
        # We type-ignore the executor assignment because the parent expects
        # a sync Callable, but we document that it must be async.
        super().__init__(
            executor=executor,  # type: ignore
            max_rollbacks=max_rollbacks,
            detectors=detectors,
            goal_drift_threshold=goal_drift_threshold,
            store=store,
            constraints=constraints,
            on_rollback=on_rollback,
            embedding_backend=embedding_backend,
            step_parser=step_parser,
            max_steps=max_steps,
            verbose=verbose,
        )
        self.async_executor: Callable[[str], Awaitable[str]] = executor

    async def run(self, goal: str) -> AgentRunResult:  # type: ignore[override]
        logger.info("Starting async agent run with goal: %s", goal[:100])

        goal_embedding = self.embedder.embed(goal)

        step_counter = 0
        rollback_count = 0
        rollback_history: list[RollbackResult] = []
        current_prompt = goal
        injection_context = ""
        last_output = ""

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
        prev_state = initial_state

        while step_counter < self.max_steps:
            step_counter += 1
            logger.debug("Step %d", step_counter)

            prompt = current_prompt
            if injection_context:
                prompt = f"{injection_context}\n\n{current_prompt}"
                injection_context = ""

            try:
                # AWAIT the executor
                raw_output = await self.async_executor(prompt)
            except Exception as exc:
                raw_output = f"OBSERVATION: Error executing agent: {exc}"
                logger.error("Executor raised: %s", exc)

            step_result = self.step_parser(raw_output)
            last_output = step_result.observation or step_result.raw_output

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
                metadata={"raw_output": raw_output},
            )

            self.store.save(state)
            self.graph.add_transition(prev_state, state)

            if step_result.done:
                logger.info("Agent signalled completion at step %d", step_counter)
                return AgentRunResult(
                    success=True,
                    final_output=last_output,
                    total_steps=step_counter,
                    rollbacks_performed=rollback_count,
                    causal_graph_mermaid=self.graph.visualize(),
                    rollback_history=rollback_history,
                )

            history = self.store.get_all()
            signal = self._detector.detect(state, history[:-1])

            if signal.detected:
                logger.warning(
                    "Divergence detected at step %d: %s (confidence=%.2f)",
                    step_counter, signal.reason, signal.confidence,
                )

                if rollback_count >= self.max_rollbacks:
                    logger.error("Max rollbacks (%d) exceeded", self.max_rollbacks)
                    return AgentRunResult(
                        success=False,
                        final_output=f"Max rollbacks exceeded. Last output: {last_output}",
                        total_steps=step_counter,
                        rollbacks_performed=rollback_count,
                        causal_graph_mermaid=self.graph.visualize(),
                        rollback_history=rollback_history,
                    )

                self.graph.mark_node_diverged(step_counter)

                safe_state = self.graph.get_safe_rollback_point(
                    step_counter,
                    goal_sim_threshold=self.goal_drift_threshold,
                )
                if safe_state is None:
                    safe_state = initial_state
                    logger.warning("No safe rollback point found, falling back to S0")

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

                if self.on_rollback:
                    self.on_rollback(result)

                logger.info(
                    "Rolled back to step %d (%d steps undone)",
                    safe_state.step_id, result.steps_undone,
                )

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

                prev_state = safe_state
                current_prompt = goal
                continue

            prev_state = state
            current_prompt = goal

        logger.warning("Max steps (%d) exceeded", self.max_steps)
        return AgentRunResult(
            success=False,
            final_output=f"Max steps exceeded. Last output: {last_output}",
            total_steps=step_counter,
            rollbacks_performed=rollback_count,
            causal_graph_mermaid=self.graph.visualize(),
            rollback_history=rollback_history,
        )
