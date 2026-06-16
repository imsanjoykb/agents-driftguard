import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# State variables to keep track of simulation recovery paths
call_count = 0
logs_fixed = False
metrics_fixed = False
report_fixed = False

def data_pipeline_executor(prompt: str) -> str:
    global call_count, logs_fixed, metrics_fixed, report_fixed
    call_count += 1
    time.sleep(1.5)  # Slow down for dashboard observation
    
    # Check if context tells us we rolled back and what resource was warned
    if "ROLLBACK" in prompt:
        if "raw_logs.json" in prompt:
            logs_fixed = True
        elif "metrics" in prompt:
            metrics_fixed = True
        elif "report.csv" in prompt:
            report_fixed = True

    print(f"  [Agent Executing Call #{call_count}] (LogsFixed={logs_fixed}, MetricsFixed={metrics_fixed}, ReportFixed={report_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Query API Endpoints
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Let's query log server coordinates.\n"
            "ACTION: query_api\n"
            "OBSERVATION: Log server endpoints resolved."
        )
        
    # -------------------------------------------------------------
    # Step 2: Download raw_logs.json
    # -------------------------------------------------------------
    elif call_count == 2:
        if not logs_fixed:
            return (
                "THOUGHT: Downloading raw JSON logs to raw_logs.json.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to raw_logs.json successfully (warning: corrupted characters in log stream)"
            )
        else:
            return (
                "THOUGHT: Re-downloading raw JSON logs with corrected encoding format.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to raw_logs.json successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Parse raw_logs.json
    # -------------------------------------------------------------
    elif call_count == 3:
        if not logs_fixed:
            # Divergence 1: Bad raw_logs.json format
            return (
                "THOUGHT: Parsing raw_logs.json.\n"
                "ACTION: parse_json\n"
                "OBSERVATION: Error: ValueError: Invalid control character in raw_logs.json"
            )
        else:
            return (
                "THOUGHT: Parsing logs with correct encoding.\n"
                "ACTION: parse_json\n"
                "OBSERVATION: Parsed 100 log lines successfully."
            )
            
    # -------------------------------------------------------------
    # Step 4: Write clean logs to metrics database table
    # -------------------------------------------------------------
    elif call_count == 4:
        if not metrics_fixed:
            return (
                "THOUGHT: Writing clean stats to metrics DB table.\n"
                "ACTION: execute_query\n"
                "OBSERVATION: saved to metrics successfully (warning: database pool connection locks active)"
            )
        else:
            return (
                "THOUGHT: Re-writing stats to metrics DB table with increased timeout configuration.\n"
                "ACTION: execute_query\n"
                "OBSERVATION: saved to metrics successfully"
            )
            
    # -------------------------------------------------------------
    # Step 5: Query weather conditions (unrelated step)
    # -------------------------------------------------------------
    elif call_count == 5:
        return (
            "THOUGHT: Gathering current weather status.\n"
            "ACTION: query_weather\n"
            "OBSERVATION: Weather conditions: Raining, 18C"
        )
        
    # -------------------------------------------------------------
    # Step 6: Generate intermediate report.csv
    # -------------------------------------------------------------
    elif call_count == 6:
        if not report_fixed:
            return (
                "THOUGHT: Exporting logs stats to CSV report.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to report.csv successfully (warning: missing file headers)"
            )
        else:
            return (
                "THOUGHT: Re-exporting report.csv with valid column headers.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to report.csv successfully"
            )
            
    # -------------------------------------------------------------
    # Step 7: Read metrics database table (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 7:
        if not metrics_fixed:
            # Divergence 2: metrics DB pool connection timeout
            return (
                "THOUGHT: Querying historical averages from metrics.\n"
                "ACTION: execute_query\n"
                "OBSERVATION: Error: sqlite3.OperationalError: database table is locked: metrics"
            )
        else:
            return (
                "THOUGHT: Querying historical metrics averages successfully.\n"
                "ACTION: execute_query\n"
                "OBSERVATION: Query successful: average latency retrieved."
            )
            
    # -------------------------------------------------------------
    # Step 8: Build report PDF from report.csv (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 8:
        if not report_fixed:
            # Divergence 3: report.csv header syntax error
            return (
                "THOUGHT: Building PDF compilation from report.csv.\n"
                "ACTION: compile_pdf\n"
                "OBSERVATION: Error: KeyError: Column 'Timestamp' missing in report.csv"
            )
        else:
            return (
                "THOUGHT: Compiling PDF report from report.csv successfully.\n"
                "ACTION: compile_pdf\n"
                "OBSERVATION: saved to analytics.pdf successfully"
            )
            
    # -------------------------------------------------------------
    # Step 9: Save PDF locally
    # -------------------------------------------------------------
    elif call_count == 9:
        return (
            "THOUGHT: Saving the compiled PDF locally.\n"
            "ACTION: write_file\n"
            "OBSERVATION: saved to analytics.pdf successfully"
        )
        
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Trajectory completed successfully. DriftGuard resolved 3 distinct dependency failures:\n"
            "  1. Step 3 failed on raw_logs.json -> Rolled back to step 1 (undid S2).\n"
            "  2. Step 7 failed on DB table metrics -> Rolled back to step 3 (undid S4, S5, S6).\n"
            "  3. Step 8 failed on report.csv -> Rolled back to step 5 (undid S6, S7)."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=data_pipeline_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting 10-step Multi-Rollback Data Pipeline Demo! Open http://localhost:8050 in your browser.")
result = wrapper.run(goal="Process logs, insert to database, and compile PDF report.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
