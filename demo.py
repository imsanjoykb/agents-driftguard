"""
DriftGuard Demo — Run this to see causal rollback in action!

Usage:
    python demo.py                  # Basic demo with mock agent
    python demo.py --scenario 2     # Agent stuck in a loop
    python demo.py --scenario 3     # Custom constraint violation
    python demo.py --verbose        # See detailed logs
"""

import argparse
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from agents_driftguard import AgentUndoWrapper


# ─────────────────────────────────────────────────────────────────────
# Scenario 1: Tool Failure → Rollback → Recovery
# ─────────────────────────────────────────────────────────────────────

def scenario_tool_failure(verbose: bool = False) -> None:
    """An agent researching a topic hits a tool error, rolls back, and recovers."""

    print("=" * 60)
    print("🔬 SCENARIO 1: Tool Failure Recovery")
    print("=" * 60)
    print()
    print("Setup: An agent is researching AI papers.")
    print("  - Steps 1-2: search + analyze (succeed)")
    print("  - Step 3: write_file (FAILS with permission error)")
    print("  - After rollback → Step 4: outputs answer directly")
    print()

    call_count = 0

    def research_agent(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        # Show what the agent sees
        has_rollback = "ROLLBACK" in prompt
        if has_rollback:
            print(f"  📨 Agent sees ROLLBACK CONTEXT in prompt")

        if call_count == 1:
            print(f"  🔍 Step {call_count}: Agent searches for AI papers...")
            return (
                "THOUGHT: I need to search for recent AI papers.\n"
                "ACTION: search_papers\n"
                "OBSERVATION: Found 12 papers on transformer architectures "
                "published in 2024, including 'Attention Is Still All You Need'."
            )
        elif call_count == 2:
            print(f"  📊 Step {call_count}: Agent analyzes the results...")
            return (
                "THOUGHT: Let me analyze the key findings.\n"
                "ACTION: analyze\n"
                "OBSERVATION: Key themes: efficiency improvements (3 papers), "
                "multimodal fusion (4 papers), long-context handling (5 papers)."
            )
        elif call_count == 3:
            print(f"  💥 Step {call_count}: Agent tries to write a file (THIS WILL FAIL)...")
            return (
                "THOUGHT: I'll save the summary to a file.\n"
                "ACTION: write_file\n"
                "OBSERVATION: Error: PermissionError - Cannot write to /reports/summary.md"
            )
        else:
            print(f"  ✅ Step {call_count}: Agent outputs answer directly (different strategy!)")
            return (
                "THOUGHT: Writing failed, I'll just output directly.\n"
                "FINAL ANSWER: Based on 12 papers analyzed, the top 3 trends in "
                "AI research are: (1) Efficiency improvements in transformers, "
                "(2) Multimodal fusion architectures, (3) Long-context handling."
            )

    wrapper = AgentUndoWrapper(
        executor=research_agent,
        max_rollbacks=3,
        detectors=["tool_failure"],      # Only watch for tool errors
        embedding_backend="dummy",       # No API key needed
        store="memory",
        verbose=verbose,
    )

    print("▶ Running agent...\n")
    result = wrapper.run(goal="Research recent AI papers and summarize trends")

    _print_result(result)
    _print_graph(wrapper)


# ─────────────────────────────────────────────────────────────────────
# Scenario 2: Loop Detection → Rollback
# ─────────────────────────────────────────────────────────────────────

def scenario_loop_detection(verbose: bool = False) -> None:
    """An agent gets stuck repeating the same action."""

    print("=" * 60)
    print("🔄 SCENARIO 2: Loop Detection")
    print("=" * 60)
    print()
    print("Setup: An agent keeps retrying the same failing search.")
    print("  - Loop detector catches repeated state hashes.")
    print("  - After rollback → Agent tries a different approach.")
    print()

    call_count = 0

    def looping_agent(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        has_rollback = "ROLLBACK" in prompt
        if has_rollback:
            print(f"  📨 Agent sees ROLLBACK CONTEXT — will try differently")

        if call_count <= 3 and not has_rollback:
            # Keep doing the exact same thing (triggers loop detection)
            print(f"  🔁 Step {call_count}: Agent searches 'quantum computing' (same query every time)...")
            return (
                "THOUGHT: Let me search for information.\n"
                "ACTION: search\n"
                "OBSERVATION: No results found for quantum computing applications."
            )
        else:
            print(f"  ✅ Step {call_count}: Agent tries a different search query!")
            return (
                "THOUGHT: Previous searches failed. Let me try a broader query.\n"
                "FINAL ANSWER: Quantum computing is advancing in error correction, "
                "with Google's Willow chip demonstrating below-threshold error rates."
            )

    wrapper = AgentUndoWrapper(
        executor=looping_agent,
        max_rollbacks=3,
        detectors=["loop"],              # Only watch for loops
        embedding_backend="dummy",
        store="memory",
        verbose=verbose,
    )

    print("▶ Running agent...\n")
    result = wrapper.run(goal="Find information about quantum computing breakthroughs")

    _print_result(result)
    _print_graph(wrapper)


# ─────────────────────────────────────────────────────────────────────
# Scenario 3: Custom Constraint Violation
# ─────────────────────────────────────────────────────────────────────

def scenario_constraint(verbose: bool = False) -> None:
    """An agent violates a business rule (budget constraint)."""

    print("=" * 60)
    print("🚧 SCENARIO 3: Custom Constraint Violation")
    print("=" * 60)
    print()
    print('Setup: A constraint says: "observation must not contain `delete`".')
    print("  - Agent tries to delete something → constraint fires.")
    print("  - After rollback → Agent takes a safer action.")
    print()

    call_count = 0

    def risky_agent(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        has_rollback = "ROLLBACK" in prompt

        if call_count == 1:
            print(f"  📋 Step {call_count}: Agent lists files...")
            return (
                "ACTION: list_files\n"
                "OBSERVATION: Found 3 files: report.md, data.csv, temp.log"
            )
        elif call_count == 2 and not has_rollback:
            print(f"  ⚠️  Step {call_count}: Agent tries to DELETE a file!")
            return (
                "ACTION: cleanup\n"
                "OBSERVATION: About to delete temp.log and data.csv"
            )
        else:
            print(f"  ✅ Step {call_count}: Agent archives instead of deleting!")
            return (
                "ACTION: archive\n"
                "FINAL ANSWER: Archived temp.log to /archive/. "
                "Kept data.csv as it may be needed. Report is clean."
            )

    # Custom constraint: observation must never contain "delete"
    no_delete_constraint = lambda state: "delete" not in state.observation.lower()

    wrapper = AgentUndoWrapper(
        executor=risky_agent,
        max_rollbacks=3,
        detectors=["constraint"],
        embedding_backend="dummy",
        store="memory",
        constraints=[no_delete_constraint],
        verbose=verbose,
    )

    print("▶ Running agent...\n")
    result = wrapper.run(goal="Clean up the project directory safely")

    _print_result(result)
    _print_graph(wrapper)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _print_result(result) -> None:
    """Pretty-print the AgentRunResult."""
    print()
    print("─" * 60)
    print("📋 RESULT")
    print("─" * 60)
    status = "✅ SUCCESS" if result.success else "❌ FAILED"
    print(f"  Status:     {status}")
    print(f"  Output:     {result.final_output[:100]}")
    print(f"  Steps:      {result.total_steps}")
    print(f"  Rollbacks:  {result.rollbacks_performed}")

    if result.rollback_history:
        print()
        print("  📜 Rollback History:")
        for i, rb in enumerate(result.rollback_history):
            print(f"     #{i+1}: Rolled back to step {rb.rolled_back_to} "
                  f"({rb.steps_undone} steps undone)")
            for event in rb.events:
                print(f"         → {event.event_type}: {event.details}")


def _print_graph(wrapper) -> None:
    """Print the Mermaid causal graph."""
    print()
    print("─" * 60)
    print("🗺️  CAUSAL GRAPH (Mermaid)")
    print("─" * 60)
    mermaid = wrapper.graph.visualize()
    # Indent for readability
    for line in mermaid.split("\n"):
        print(f"  {line}")
    print()
    print("💡 Paste the above into https://mermaid.live to visualize!")
    print()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard interactive demo")
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Run a specific scenario (1=tool_failure, 2=loop, 3=constraint). "
             "Default: run all.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    scenarios = {
        1: scenario_tool_failure,
        2: scenario_loop_detection,
        3: scenario_constraint,
    }

    if args.scenario == 0:
        # Run all
        for i, fn in scenarios.items():
            fn(verbose=args.verbose)
            if i < len(scenarios):
                print("\n" + "═" * 60 + "\n")
    elif args.scenario in scenarios:
        scenarios[args.scenario](verbose=args.verbose)
    else:
        print(f"Unknown scenario {args.scenario}. Choose 1, 2, or 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
