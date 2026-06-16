import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# 1. A simulated compiler build agent
call_count = 0
def compiler_agent_executor(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # We check if a rollback context warning is present in the prompt
    if "ROLLBACK" in prompt:
        # Recovery flow: correct file syntax
        if call_count == 3:
            return (
                "THOUGHT: Correcting brackets syntax in utils.py.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to utils.py successfully with correct syntax"
            )
        elif call_count == 4:
            return (
                "THOUGHT: Attempting build compile sequence again.\n"
                "ACTION: compile_build\n"
                "OBSERVATION: Build success! All modules compiled successfully."
            )
        else:
            return (
                "FINAL ANSWER: Compiler build pipeline completed successfully. "
                "DriftGuard detected a syntax error at step 5 in 'utils.py', "
                "rolled back directly to step 1 (bypassing steps 3 and 4), allowing us to fix 'utils.py'."
            )
    else:
        # Default failing flow
        if call_count == 1:
            return (
                "THOUGHT: Reading project specifications.\n"
                "ACTION: read_specs\n"
                "OBSERVATION: Specs parsed successfully."
            )
        elif call_count == 2:
            return (
                "THOUGHT: Writing utility functions.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to utils.py successfully (contains syntax bracket error)"
            )
        elif call_count == 3:
            return (
                "THOUGHT: Writing main loop code.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to main.py successfully"
            )
        elif call_count == 4:
            return (
                "THOUGHT: Optimizing application assets.\n"
                "ACTION: optimize_images\n"
                "OBSERVATION: Assets optimized."
            )
        else:
            return (
                "THOUGHT: Compiling project build.\n"
                "ACTION: compile_build\n"
                "OBSERVATION: Error compiling build: SyntaxError: unexpected EOF in utils.py"
            )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=compiler_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=3,
)

print("🚀 Starting Compiler Pipeline Fix demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Compile and build code modules.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
