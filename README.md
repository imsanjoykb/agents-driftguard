<p align="center">
  <h1 align="center">🛡️ DriftGuard</h1>
  <p align="center"><strong>A State-Aware Rollback and Trajectory-Replanning Framework for Autonomous LLM Agents</strong></p>
  <p align="center">
    <a href="https://github.com/imsanjoykb/agents-driftguard/actions/workflows/test.yml"><img src="https://github.com/imsanjoykb/agents-driftguard/actions/workflows/test.yml/badge.svg" alt="CI Status"></a>
    <a href="https://pypi.org/project/agents-driftguard/"><img src="https://img.shields.io/pypi/v/agents-driftguard.svg" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/agents-driftguard/"><img src="https://img.shields.io/pypi/pyversions/agents-driftguard.svg" alt="Python Versions"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black"></a>
  </p>
  <p align="center">
    <a href="#key-features">Features</a> •
    <a href="#the-problem-and-solution">The Concept</a> •
    <a href="#installation">Installation</a> •
    <a href="#quickstart">Quickstart Tutorial</a> •
    <a href="#how-it-works">How it Works</a> •
    <a href="#advanced-configurations">Advanced Configurations</a> •
    <a href="#citations">Research & Citation</a>
  </p>
</p>

---

## What is DriftGuard?

**DriftGuard** is a system-level execution wrapper for autonomous Large Language Model (LLM) agents. It acts as a safety layer around any LLM agent executor, continuously monitoring trajectories, detecting mid-chain divergences (such as infinite loops, tool failures, or goal drift), automatically executing database-style **reversion hooks** to restore the physical environment, and replanning agent prompts with full context of what failed.

Instead of terminating your agent runs or restarting them from scratch when an error occurs, DriftGuard recovers execution mid-chain, saving you tokens, time, and context window space.

---

## Key Features

* 🔄 **State-Aware Rollbacks**: Snapshots conversational memory, reasoning traces, and resource states at every step.
* 🛠️ **Reversible Tool Registry**: Register compensating (undo) hooks to reverse physical environment side effects (e.g. deleting files, reverting DB queries) on failure.
* ⛓️ **Dependency-Aware Rollbacks**: Tracks data dependencies ($R_{in}$, $R_{out}$) to pinpoint the exact ancestor node that introduced a defect and roll back directly to it.
* 🚨 **Ensemble Divergence Detectors**: Pluggable monitors that evaluate loop patterns, token limits, linguistic contradictions, output quality, and goal similarity.
* 📊 **Live Telemetry Dashboard**: Launch a local, zero-dependency dashboard server to watch agent runs, branching causal DAGs, and rollbacks in real time.
* 🔌 **Framework Agnostic**: Includes native adapters for **LangChain** and **OpenAI Function Calling**.

---

## The Problem and Solution

### The Problem
Agents executing multi-step tasks often fail mid-trajectory (e.g., executing a bash command with syntax errors). Existing cognitive self-correction strategies (like Tree of Thoughts or Reflexion) try to solve this by re-prompting the LLM. 

However, in real-world embodied settings, **tool actions mutate the physical environment**. If the agent creates a broken codebase structure or alters a database table, merely backtracking the prompt does not revert the physical environment. The agent remains blocked by its own side-effects.

### The Solution
DriftGuard introduces system-level causal transactions to agent execution loops:

```
S0 ──► S1 ──► S2 ──► S3 (💥 environment error / loop)
               ▲
               │ [Rollback: Truncates stores & reverses S3 -> S2 side-effects]
               │
               S2 ──► S4 (✅ Success path with failure context injected)
```

---

## Installation

Install via pip:

```bash
pip install agents-driftguard
```

### Optional Backends
Customize installations based on your backend environment:

```bash
# To use local sentence-transformer embeddings for Goal Drift detection
pip install agents-driftguard[embeddings]

# To use OpenAI API embeddings & Function Calling Adapter
pip install agents-driftguard[openai]

# To wrap LangChain Agent Executors
pip install agents-driftguard[langchain]

# For development (pytest, formatting, linting)
pip install agents-driftguard[dev]
```

---

## Quickstart Tutorial

This tutorial demonstrates how to wrap a simple agent, register a reversible filesystem tool, and handle errors automatically.

### Step 1: Wrap Your Agent Executor
Create your agent execution logic (which accepts a prompt and returns a thought/action response) and wrap it with DriftGuard:

```python
import os
from agents_driftguard import AgentUndoWrapper

# A mock agent executor
def my_agent(prompt: str) -> str:
    # If the LLM has received rollback feedback context, it corrects its action
    if "ROLLBACK" in prompt:
        return "ACTION: write_file\nOBSERVATION: saved to config.json with correct syntax\nDONE"
        
    # Standard linear path that introduces an error
    return "ACTION: write_file\nOBSERVATION: Error compiling build due to SyntaxError in config.json"

# Wrap it with DriftGuard
wrapper = AgentUndoWrapper(
    executor=my_agent,
    max_rollbacks=3,
    detectors=["tool_failure", "loop"],
    embedding_backend="dummy"
)
```

### Step 2: Register Reversible Tool Side-Effects
Decorate your state-mutating actions with the `@reversible` decorator and register an undo function. DriftGuard will execute the undo hook in reverse chronological order during rollbacks:

```python
from agents_driftguard import reversible

# Define compensating (undo) logic
def undo_write(args, observation):
    path = args.get("path")
    if os.path.exists(path):
        os.remove(path)
        print(f"🔄 Reverted environment: Deleted {path}")

# Decorate the tool
@reversible(undo_fn=undo_write, tool_name="write_file")
def write_file(path: str, content: str) -> str:
    with open(path, "w") as f:
        f.write(content)
    return f"Saved content to {path}"
```

### Step 3: Run with Automatic Protection
Execute the wrapper. DriftGuard will trace the execution, detect the compile error in step 1, trigger the `undo_write` function, inject the failure context into the next prompt, and re-run:

```python
result = wrapper.run(goal="Create a config.json configuration module")

print(f"Goal Succeeded: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Answer: {result.final_output}")
```

---

## How it Works

### 1. Execution Wrapper Loop
During `wrapper.run(goal)`:
1. DriftGuard initializes the run and embeds the goal.
2. The Agent Executor is called.
3. The raw output is parsed into a standard `StepResult`.
4. An `AgentState` snapshot is saved, and a transition node is added to the `CausalGraph`.
5. Active `DivergenceDetectors` evaluate the transition.
6. If divergence is detected:
   * The node is marked as diverged.
   * The graph identifies the safest ancestor rollback state.
   * The `RollbackController` executes registered undo hooks for all intermediate steps in reverse order, restoring the environment.
   * The `Replanner` injects a summary of the failure path into the prompt context.
   * Execution resumes from the safe state.
7. If no divergence is found, execution proceeds normally.

### 2. Pluggable Divergence Detectors

DriftGuard evaluates execution health via a suite of configurable heuristics:

| Detector | Category | Metric / Heuristic |
|---|---|---|
| **`GoalDriftDetector`** | Semantic | Cosine similarity between goal and state text embeddings. |
| **`LoopDetector`** | Sequence | Matches SHA-256 state hashes in a sliding window to stop infinite loops. |
| **`ToolFailureDetector`** | Regex | Scans tool observations for syntax tracebacks, HTTP codes (4xx/5xx), or exceptions. |
| **`HallucinationDetector`** | Linguistic | Flags contradictions (e.g. "success" vs "failed") regarding overlapping subjects. |
| **`TokenBudgetDetector`** | Resource | Tracks token consumption to prevent context-window collapse. |
| **`OutputQualityDetector`** | Content | Checks for empty outputs or high word repetition ratios. |
| **`ConstraintChecker`** | Custom | Evaluates custom developer-defined Python check lambdas. |
| **`LLMJudge`** | AI Eval | Uses an LLM to rate action alignment against the target goal. |

---

## Advanced Configurations

### Persistent SQLite State Store
For production environments, swap the default in-memory state tracker with a persistent SQLite store:

```python
from agents_driftguard.store.sqlite import SQLiteStateStore

store = SQLiteStateStore(db_path="agent_telemetry.db")
wrapper = AgentUndoWrapper(executor=my_agent, store=store)
```

### Async Agent Support
Natively wrap asynchronous agent execution setups:

```python
import asyncio
from agents_driftguard.async_wrapper import AsyncAgentUndoWrapper

async def my_async_agent(prompt: str) -> str:
    return await async_llm.generate(prompt)

async def main():
    wrapper = AsyncAgentUndoWrapper(executor=my_async_agent)
    result = await wrapper.arun(goal="Run async task")
    print(result.success)

asyncio.run(main())
```

### Live Telemetry Dashboard
Launch the zero-dependency web dashboard to inspect agent trajectories, inspect branching causal DAGs, check rollback timeline logs, and track token consumption metrics:

```python
from agents_driftguard.dashboard.app import start_dashboard_server

# Start server in the background
start_dashboard_server(port=8050)

# Telemetry will now stream to http://localhost:8050 as you run wrapper.run()
```

---

## Framework Integrations

### LangChain Integration
Wrap a LangChain `AgentExecutor` using the built-in adapter:

```python
from langchain.agents import AgentExecutor
from agents_driftguard.integrations.langchain import LangChainAdapter

executor = AgentExecutor(agent=..., tools=...)
adapter = LangChainAdapter(executor)

wrapper = AgentUndoWrapper(
    executor=adapter,
    step_parser=adapter.parse_step,
)
```

### OpenAI Function Calling Integration
Wrap chat completions leveraging OpenAI tool calls:

```python
from openai import OpenAI
from agents_driftguard.integrations.openai import OpenAIFunctionCallingAdapter

client = OpenAI()
adapter = OpenAIFunctionCallingAdapter(
    client=client,
    model="gpt-4o",
    tools=[...],
    tool_executor=my_tool_runner,
)

wrapper = AgentUndoWrapper(
    executor=adapter,
    step_parser=adapter.parse_step,
)
```

---

## Citations

If you use this framework in your research, please cite our paper:

```bibtex
@article{driftguard2026,
  title={DriftGuard: A State-Aware Rollback and Trajectory-Replanning Framework for Autonomous LLM Agents},
  author={Anonymous Authors},
  journal={Preprint / Working Paper},
  year={2026}
}
```

The mathematical models, system design block diagrams, and SWE-bench/WebArena evaluation results are documented in:
* [DriftGuard_Research_Paper_v3.docx](file:///d:/Project/agentundo/paper/DriftGuard_Research_Paper_v3.docx) (Word/DOCX draft)
* [driftguard_paper.md](file:///d:/Project/agentundo/paper/driftguard_paper.md) (Markdown master copy)

---

## License

MIT
