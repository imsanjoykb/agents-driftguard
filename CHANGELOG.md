# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-18

### Added
- **Dependency-Aware Rollback Traversal**: Introduced input/output resource tracking ($R_{in}$, $R_{out}$) on `AgentState`. The `CausalGraph` now walks backward to pinpoint and rollback directly to the exact step that modified a failed file, database table, or configuration.
- **Extended Divergence Detectors**: Added three new pluggable detectors:
  - `OutputQualityDetector`: Checks for empty outputs or high word repetition ratios.
  - `TokenBudgetDetector`: Tracks whitespace token estimations to prevent context window collapse.
  - `HallucinationDetector`: Scans for semantic contradictions between current and past observations on the same subject.
- **Academic Telemetry Support**: Added diagrams (`causal_diagram.png` and `system_architecture.png`) and comprehensive paper compilation templates in the `paper/` directory (now ignored by Git).

### Changed
- **Linter & Code Cleanup**: Cleaned up docstrings, lint warnings, and simplified code styling across `core/`, `integrations/`, and `store/` packages.
- **Documentation Overhaul**: Completely rewrote the `README.md` to meet premium open-source package standards, adding quickstart tutorials, advanced settings (SQLite, async wrapper, telemetry dashboard), and citation BibTeX references.

### Fixed
- **CI Test Port Resolution**: Fixed a connection error in `test_dashboard.py` by replacing the `localhost` lookup with `127.0.0.1`. This resolves DNS resolution failures on GitHub Action Ubuntu runners where `localhost` resolves to IPv6 loopback (`::1`) while the server binds to IPv4.

---

## [1.0.0] — 2026-06-15

### Added
- **AgentUndoWrapper**: Core execution wrapper managing the agent lifecycle, tracking states, and coordinating rollbacks.
- **CausalGraph**: NetworkX-backed Directed Acyclic Graph (DAG) representing realizing transitions and trajectory branches.
- **Divergence Heuristics**: Pluggable sub-detector framework supporting:
  - `GoalDriftDetector` (semantic embedding cosine similarity).
  - `ToolFailureDetector` (regex observation pattern matcher).
  - `LoopDetector` (state hashing over sliding windows).
  - `ConstraintChecker` (custom validation lambdas).
  - `LLMJudge` (LLM-as-a-judge scoring).
- **Environmental Restoration**: Main loop rollback controller executing registered tool-compensating hooks in reverse chronological order via the `@reversible` decorator.
- **Adapters**: Concrete adapters for LangChain (`LangChainAdapter`) and OpenAI Function Calling (`OpenAIFunctionCallingAdapter`).
- **Telemetry Server**: Live Dashboards tracking trajectory timelines, causal graphs, and token usages.
- **State Store Backends**: Standard `InMemoryStateStore` and `SQLiteStateStore` backends for telemetry persistence.
