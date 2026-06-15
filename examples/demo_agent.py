import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# 1. A simulated agent that fails on step 3, then recovers on retry
call_count = 0
def mock_agent_executor(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5) # Slow it down so we can watch the dashboard!
    
    if call_count == 1:
        return "ACTION: search\nOBSERVATION: Querying database..."
    elif call_count == 2:
        return "ACTION: process_data\nOBSERVATION: Found 5 records."
    elif call_count == 3:
        # Simulate a crash/error!
        return "ACTION: write_file\nOBSERVATION: Error 500: Permission Denied"
    elif call_count == 4:
        # After rollback, it will try a different strategy
        return "ACTION: write_to_stdout\nOBSERVATION: Wrote records to stdout."
    else:
        return "FINAL ANSWER: Trajectory completed successfully. DriftGuard detected a tool failure (Permission Denied) at step 3 when writing to a file. Reverting the agent to the last safe state (step 2) allowed it to replan, redirect the output to stdout at step 4, and complete the run safely."

# 2. Start the live dashboard in the background
start_dashboard_server(port=8050)

# 3. Wrap our agent
wrapper = AgentUndoWrapper(
    executor=mock_agent_executor,
    detectors=["tool_failure"], # Look for the "Error 500"
    max_rollbacks=3,
)

print("🚀 Starting agent run! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Process user records and save them to a secure destination.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
