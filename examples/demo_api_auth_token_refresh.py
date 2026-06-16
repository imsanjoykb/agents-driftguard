import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# 1. A simulated network API authentication agent
call_count = 0
def net_agent_executor(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # We check if a rollback context warning is present in the prompt
    if "ROLLBACK" in prompt:
        # Recovery flow: write correct token
        if call_count == 3:
            return (
                "THOUGHT: Requesting renewed API token and writing config file.\n"
                "ACTION: write_token\n"
                "OBSERVATION: saved to access_token.json successfully"
            )
        elif call_count == 4:
            return (
                "THOUGHT: Connecting to remote endpoint using the fresh token.\n"
                "ACTION: authenticate_api\n"
                "OBSERVATION: Authentication Successful! API session established."
            )
        else:
            return (
                "FINAL ANSWER: Network API Authentication completed successfully. "
                "DriftGuard detected an invalid token at step 4 on 'access_token.json', "
                "rolled back directly to step 1 (bypassing the unrelated latency check), allowing us to renew the token."
            )
    else:
        # Default failing flow
        if call_count == 1:
            return (
                "THOUGHT: Checking network interface parameters.\n"
                "ACTION: check_net\n"
                "OBSERVATION: Network active."
            )
        elif call_count == 2:
            return (
                "THOUGHT: Storing access token.\n"
                "ACTION: write_token\n"
                "OBSERVATION: saved to access_token.json successfully (token is expired/invalid)"
            )
        elif call_count == 3:
            return (
                "THOUGHT: Measuring connection latency.\n"
                "ACTION: latency_check\n"
                "OBSERVATION: Connection Latency: 15ms"
            )
        else:
            return (
                "THOUGHT: Verifying API credentials with server.\n"
                "ACTION: authenticate_api\n"
                "OBSERVATION: Error: AuthenticationError: Invalid token read from access_token.json"
            )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=net_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=3,
)

print("🚀 Starting API Authentication Token Refresh demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Authenticate API connection.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
