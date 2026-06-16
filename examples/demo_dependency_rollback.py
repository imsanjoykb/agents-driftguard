import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# 1. A simulated agent that exhibits a file dependency issue
call_count = 0
def mock_agent_executor(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5) # Slow it down so we can observe the dashboard live!
    
    if call_count == 1:
        return (
            "THOUGHT: I need to query the database first.\n"
            "ACTION: search\n"
            "OBSERVATION: Querying database: found raw input parameters."
        )
    elif call_count == 2:
        return (
            "THOUGHT: Writing raw inputs to the CSV report.\n"
            "ACTION: write_file\n"
            "ARGS: {'path': 'data.csv'}\n"
            "OBSERVATION: saved to data.csv successfully"
        )
    elif call_count == 3:
        return (
            "THOUGHT: Performing some unrelated average statistics calculation.\n"
            "ACTION: compute_statistics\n"
            "OBSERVATION: Calculated summary: mean=24.5, std=1.2"
        )
    elif call_count == 4:
        return (
            "THOUGHT: Reading the CSV to compute final output.\n"
            "ACTION: read_file\n"
            "ARGS: {'path': 'data.csv'}\n"
            "OBSERVATION: Error: Permission denied reading data.csv"
        )
    elif call_count == 5:
        # After rollback to step 1 (preceding step 2), the agent replans and uses a local fallback path
        return (
            "THOUGHT: The write to data.csv failed. I will write to a local fallback path data_local.csv instead.\n"
            "ACTION: write_file\n"
            "ARGS: {'path': 'data_local.csv'}\n"
            "OBSERVATION: saved to data_local.csv successfully"
        )
    elif call_count == 6:
        return (
            "THOUGHT: Reading local fallback to verify.\n"
            "ACTION: read_file\n"
            "ARGS: {'path': 'data_local.csv'}\n"
            "OBSERVATION: Read 5 valid rows: data verified"
        )
    else:
        return (
            "FINAL ANSWER: Trajectory completed successfully. DriftGuard detected a dependency failure (data.csv) at step 4. "
            "Instead of linearly rolling back to step 3 (which would not solve the file issue), DriftGuard traced the dependency back to step 2, "
            "reverted the agent to the state before step 2 (step 1), injected the rollback context, and allowed the agent to use the fallback file data_local.csv."
        )

# 2. Start the live dashboard in the background
start_dashboard_server(port=8050)

# 3. Wrap our agent
wrapper = AgentUndoWrapper(
    executor=mock_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=3,
)

print("🚀 Starting dependency-aware rollback agent run! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Process and verify user data securely.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
