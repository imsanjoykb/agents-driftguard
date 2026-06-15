# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2025-01-01

### Added

- `AgentUndoWrapper` — main API wrapping any agent executor with rollback-aware state management.
- `CausalGraph` — NetworkX-backed DAG tracking agent state transitions.
- `DivergenceDetector` — pluggable sub-detector framework (GoalDrift, ToolFailure, Loop, Constraint, LLMJudge).
- `RollbackController` — restores agent to last safe state with reversible tool support.
- `Replanner` — failure-context injection for the next prompt after rollback.
- `InMemoryStateStore` and `SQLiteStateStore` — state persistence backends.
- Integration adapters for LangChain and OpenAI function-calling agents.
- `@reversible` decorator for registering undo hooks on tools.
- Mermaid visualization of causal graphs.
- Embedding backends: sentence-transformers, OpenAI, and dummy (offline).
