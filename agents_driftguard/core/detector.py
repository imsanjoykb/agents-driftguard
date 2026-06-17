from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Any, Callable
from pydantic import BaseModel, Field
from agents_driftguard.core.state import AgentState
from agents_driftguard.utils.embeddings import cosine_similarity

class DivergenceSignal(BaseModel):

    detected: bool = False
    confidence: float = 0.0
    reason: str = ""
    diverged_at: str = ""
    recommended_rollback: str = ""
    details: str = ""
    failed_resource: str | None = None

class BaseDetector(ABC):
    @abstractmethod
    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        ...

class GoalDriftDetector(BaseDetector):

    def __init__(
        self,
        absolute_threshold: float = 0.75,
        step_drop_threshold: float = 0.2,
    ) -> None:
        self.absolute_threshold = absolute_threshold
        self.step_drop_threshold = step_drop_threshold

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        if not state.goal_embedding or not state.state_embedding:
            return DivergenceSignal()

        current_sim = cosine_similarity(state.goal_embedding, state.state_embedding)

        # Absolute threshold check
        if current_sim < self.absolute_threshold:
            return DivergenceSignal(
                detected=True,
                confidence=min(1.0, (self.absolute_threshold - current_sim) / self.absolute_threshold),
                reason="goal_drift",
                diverged_at=state.node_id,
                details=(
                    f"Goal-state cosine similarity ({current_sim:.3f}) "
                    f"below threshold ({self.absolute_threshold})"
                ),
            )

        # Step-over-step drop check
        if history:
            prev = history[-1]
            if prev.goal_embedding and prev.state_embedding:
                prev_sim = cosine_similarity(prev.goal_embedding, prev.state_embedding)
                drop = prev_sim - current_sim
                if drop > self.step_drop_threshold:
                    return DivergenceSignal(
                        detected=True,
                        confidence=min(1.0, drop / self.step_drop_threshold),
                        reason="goal_drift",
                        diverged_at=state.node_id,
                        details=(
                            f"Goal-state similarity dropped by {drop:.3f} "
                            f"in one step (threshold: {self.step_drop_threshold})"
                        ),
                    )

        return DivergenceSignal()


class ToolFailureDetector(BaseDetector):
    ERROR_PATTERNS: list[str] = [
        r"(?i)\b(error|exception|traceback|failed|failure)\b",
        r"(?i)\b(timeout|timed?\s*out)\b",
        r"\b[45]\d{2}\b",  # HTTP 4xx / 5xx
        r"(?i)\bnull\b",
        r"(?i)\bnone\b",
        r"(?i)raise\s+\w+Error",
    ]

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        obs = state.observation.strip()

        # Empty observation is suspicious
        if not obs:
            return DivergenceSignal(
                detected=True,
                confidence=0.7,
                reason="tool_failure",
                diverged_at=state.node_id,
                details="Tool returned empty observation",
            )

        # Pattern matching
        for pattern in self.ERROR_PATTERNS:
            match = re.search(pattern, obs)
            if match:
                # Extract failed resource heuristics (e.g. filename or database table)
                failed_resource = None
                file_match = re.search(r"\b([a-zA-Z0-9_\-\.\/]+\.(?:csv|db|json|txt|log|xml|py|sql|sh|yaml|yml|ini))\b", obs)
                if file_match:
                    failed_resource = file_match.group(1)
                else:
                    db_match = re.search(r"(?i)\b(?:table|relation|database)\s*[:\s]\s*['\"]?([a-zA-Z0-9_]+)['\"]?\b", obs)
                    if db_match:
                        failed_resource = db_match.group(1)

                return DivergenceSignal(
                    detected=True,
                    confidence=0.85,
                    reason="tool_failure",
                    diverged_at=state.node_id,
                    details=f"Error pattern detected in observation: '{match.group()}'",
                    failed_resource=failed_resource,
                )

        return DivergenceSignal()


class LoopDetector(BaseDetector):

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        if not history:
            return DivergenceSignal()

        recent_hashes = [s.state_hash for s in history[-self.window :]]
        if state.state_hash in recent_hashes:
            return DivergenceSignal(
                detected=True,
                confidence=0.95,
                reason="loop_detected",
                diverged_at=state.node_id,
                details=(
                    f"State hash {state.state_hash[:12]}... matches a recent state "
                    f"within the last {self.window} steps"
                ),
            )

        return DivergenceSignal()


class ConstraintChecker(BaseDetector):

    def __init__(self, constraints: list[Callable[[AgentState], bool]] | None = None) -> None:
        self.constraints: list[Callable[[AgentState], bool]] = constraints or []

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        for i, constraint in enumerate(self.constraints):
            try:
                if not constraint(state):
                    return DivergenceSignal(
                        detected=True,
                        confidence=1.0,
                        reason="constraint_violation",
                        diverged_at=state.node_id,
                        details=f"Constraint #{i} violated: {getattr(constraint, '__name__', repr(constraint))}",
                    )
            except Exception as exc:
                return DivergenceSignal(
                    detected=True,
                    confidence=0.8,
                    reason="constraint_violation",
                    diverged_at=state.node_id,
                    details=f"Constraint #{i} raised an exception: {exc}",
                )

        return DivergenceSignal()


class LLMJudge(BaseDetector):
    def __init__(
        self,
        judge_fn: Callable[[str], float] | None = None,
        threshold: float = 5.0,
    ) -> None:
        self.judge_fn = judge_fn
        self.threshold = threshold

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        if self.judge_fn is None:
            return DivergenceSignal()

        prompt = (
            f"The agent was given the goal context embedded at step 0.\n"
            f"Current step: {state.step_id}\n"
            f"Action taken: {state.action}\n"
            f"Observation: {state.observation}\n"
            f"Scratchpad: {state.scratchpad}\n\n"
            f"Does this action align with the goal? Score 0-10."
        )

        try:
            score = self.judge_fn(prompt)
            if score < self.threshold:
                return DivergenceSignal(
                    detected=True,
                    confidence=min(1.0, (self.threshold - score) / self.threshold),
                    reason="llm_judge",
                    diverged_at=state.node_id,
                    details=f"LLM judge scored {score:.1f} / 10 (threshold: {self.threshold})",
                )
        except Exception as exc:
            return DivergenceSignal(
                detected=True,
                confidence=0.5,
                reason="llm_judge",
                diverged_at=state.node_id,
                details=f"LLM judge call failed: {exc}",
            )

        return DivergenceSignal()

class OutputQualityDetector(BaseDetector):

    def __init__(
        self,
        min_length: int = 10,
        max_repetition_ratio: float = 0.5,
    ) -> None:
        self.min_length = min_length
        self.max_repetition_ratio = max_repetition_ratio

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        obs = state.observation.strip()

        # Too short
        if 0 < len(obs) < self.min_length:
            return DivergenceSignal(
                detected=True,
                confidence=0.75,
                reason="output_quality",
                diverged_at=state.node_id,
                details=(
                    f"Observation too short ({len(obs)} chars, "
                    f"minimum: {self.min_length})"
                ),
            )

        # Repetition check
        words = obs.lower().split()
        if len(words) >= 4:
            unique = set(words)
            repetition_ratio = 1.0 - (len(unique) / len(words))
            if repetition_ratio > self.max_repetition_ratio:
                return DivergenceSignal(
                    detected=True,
                    confidence=min(1.0, repetition_ratio),
                    reason="output_quality",
                    diverged_at=state.node_id,
                    details=(
                        f"High repetition ratio ({repetition_ratio:.2f}, "
                        f"threshold: {self.max_repetition_ratio})"
                    ),
                )

        return DivergenceSignal()


class TokenBudgetDetector(BaseDetector):

    def __init__(self, max_tokens: int = 10000) -> None:
        self.max_tokens = max_tokens

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:
        total_tokens = 0
        for s in history:
            total_tokens += len(s.observation.split())
            total_tokens += len(s.scratchpad.split())
        # Add current step
        total_tokens += len(state.observation.split())
        total_tokens += len(state.scratchpad.split())

        if total_tokens > self.max_tokens:
            return DivergenceSignal(
                detected=True,
                confidence=min(1.0, total_tokens / (self.max_tokens * 1.5)),
                reason="token_budget",
                diverged_at=state.node_id,
                details=(
                    f"Cumulative tokens ({total_tokens}) exceed budget "
                    f"({self.max_tokens})"
                ),
            )

        return DivergenceSignal()


class HallucinationDetector(BaseDetector):

    DEFAULT_PAIRS: list[tuple[str, str]] = [
        ("found", "not found"),
        ("success", "failed"),
        ("exists", "does not exist"),
        ("available", "unavailable"),
        ("connected", "disconnected"),
        ("completed", "not completed"),
    ]

    def __init__(
        self,
        contradiction_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        self.contradiction_pairs = contradiction_pairs or self.DEFAULT_PAIRS

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:

        if not history:
            return DivergenceSignal()

        current_obs = state.observation.lower()
        past_obs = " ".join(s.observation.lower() for s in history)

        for positive, negative in self.contradiction_pairs:
            # Current says positive, past said negative (or vice versa)
            if positive in current_obs and negative in past_obs:
                # Check it's about the same subject (simple word overlap)
                current_words = set(current_obs.split())
                for s in history:
                    if negative in s.observation.lower():
                        past_words = set(s.observation.lower().split())
                        overlap = current_words & past_words
                        # Need at least 2 common content words
                        content_overlap = {
                            w for w in overlap
                            if len(w) > 3 and w not in {positive, negative, "the", "and", "for", "that", "this", "with"}
                        }
                        if len(content_overlap) >= 2:
                            return DivergenceSignal(
                                detected=True,
                                confidence=0.8,
                                reason="hallucination",
                                diverged_at=state.node_id,
                                details=(
                                    f"Contradiction: current says '{positive}' "
                                    f"but step {s.step_id} said '{negative}' "
                                    f"(shared context: {content_overlap})"
                                ),
                            )

            if negative in current_obs and positive in past_obs:
                current_words = set(current_obs.split())
                for s in history:
                    if positive in s.observation.lower():
                        past_words = set(s.observation.lower().split())
                        overlap = current_words & past_words
                        content_overlap = {
                            w for w in overlap
                            if len(w) > 3 and w not in {positive, negative, "the", "and", "for", "that", "this", "with"}
                        }
                        if len(content_overlap) >= 2:
                            return DivergenceSignal(
                                detected=True,
                                confidence=0.8,
                                reason="hallucination",
                                diverged_at=state.node_id,
                                details=(
                                    f"Contradiction: current says '{negative}' "
                                    f"but step {s.step_id} said '{positive}' "
                                    f"(shared context: {content_overlap})"
                                ),
                            )

        return DivergenceSignal()

# Mapping of detector name → class for factory construction
DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "goal_drift": GoalDriftDetector,
    "tool_failure": ToolFailureDetector,
    "loop": LoopDetector,
    "constraint": ConstraintChecker,
    "llm_judge": LLMJudge,
    "output_quality": OutputQualityDetector,
    "token_budget": TokenBudgetDetector,
    "hallucination": HallucinationDetector,
}


class DivergenceDetector:

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        strategy: str = "any",
    ) -> None:
        self.detectors: list[BaseDetector] = detectors or []
        self.strategy = strategy

    def detect(
        self,
        state: AgentState,
        history: list[AgentState],
        **kwargs: Any,
    ) -> DivergenceSignal:

        if not self.detectors:
            return DivergenceSignal()

        signals = [d.detect(state, history, **kwargs) for d in self.detectors]
        fired = [s for s in signals if s.detected]

        if not fired:
            return DivergenceSignal()

        if self.strategy == "all":
            if len(fired) < len(self.detectors):
                return DivergenceSignal()
            return self._merge(fired, state)

        if self.strategy == "weighted":
            total_confidence = sum(s.confidence for s in fired) / len(self.detectors)
            if total_confidence < 0.5:
                return DivergenceSignal()
            merged = self._merge(fired, state)
            merged.confidence = total_confidence
            return merged

        # Default: "any"
        return self._merge(fired, state)

    @staticmethod
    def _merge(fired: list[DivergenceSignal], state: AgentState) -> DivergenceSignal:

        best = max(fired, key=lambda s: s.confidence)
        reasons = sorted({s.reason for s in fired if s.reason})
        details = "; ".join(s.details for s in fired if s.details)
        failed_res = next((s.failed_resource for s in fired if s.failed_resource), None)

        return DivergenceSignal(
            detected=True,
            confidence=best.confidence,
            reason="|".join(reasons) if len(reasons) > 1 else best.reason,
            diverged_at=state.node_id,
            recommended_rollback=best.recommended_rollback,
            details=details,
            failed_resource=failed_res,
        )
