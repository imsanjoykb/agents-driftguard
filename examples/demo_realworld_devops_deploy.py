import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# State variables to keep track of simulation recovery paths
call_count = 0
docker_fixed = False
frontend_fixed = False
schema_fixed = False

def devops_deploy_executor(prompt: str) -> str:
    global call_count, docker_fixed, frontend_fixed, schema_fixed
    call_count += 1
    time.sleep(1.5)  # Slow down for dashboard observation
    
    # Check if context tells us we rolled back and what resource was warned
    if "ROLLBACK" in prompt:
        if "docker-compose.yml" in prompt:
            docker_fixed = True
        elif "utils.js" in prompt:
            frontend_fixed = True
        elif "schema.sql" in prompt:
            schema_fixed = True

    print(f"  [Agent Executing Call #{call_count}] (DockerFixed={docker_fixed}, FrontendFixed={frontend_fixed}, SchemaFixed={schema_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Retrieve repository specifications
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Reading workspace properties.\n"
            "ACTION: read_specs\n"
            "OBSERVATION: Specs loaded."
        )
        
    # -------------------------------------------------------------
    # Step 2: Write docker-compose.yml config file
    # -------------------------------------------------------------
    elif call_count == 2:
        if not docker_fixed:
            return (
                "THOUGHT: Writing docker container setup details.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to docker-compose.yml successfully (warning: indentation syntax invalid)"
            )
        else:
            return (
                "THOUGHT: Re-writing docker container config with valid yaml properties.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to docker-compose.yml successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Write frontend code file utils.js
    # -------------------------------------------------------------
    elif call_count == 3:
        if not frontend_fixed:
            return (
                "THOUGHT: Writing web layout utility module.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to utils.js successfully (warning: referencing missing dependency module)"
            )
        else:
            return (
                "THOUGHT: Re-writing web module utils.js with bundled core dependencies.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to utils.js successfully"
            )
            
    # -------------------------------------------------------------
    # Step 4: Compile static frontend assets (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 4:
        if not frontend_fixed:
            # Divergence 1: Bad utils.js compile error
            return (
                "THOUGHT: Building web application static distribution package.\n"
                "ACTION: compile_build\n"
                "OBSERVATION: Error compiling build: ModuleNotFoundError: No module named 'utils.js'"
            )
        else:
            return (
                "THOUGHT: Frontend modules compiled correctly.\n"
                "ACTION: compile_build\n"
                "OBSERVATION: Static build compiled: 14 assets generated."
            )
            
    # -------------------------------------------------------------
    # Step 5: Start Docker Containers (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 5:
        if not docker_fixed:
            # Divergence 2: docker-compose.yml indentation syntax error
            return (
                "THOUGHT: Launching dev container instance.\n"
                "ACTION: run_docker\n"
                "OBSERVATION: Error: yaml.parser.ScannerError: mapping values are not allowed in docker-compose.yml"
            )
        else:
            return (
                "THOUGHT: Launching dev container instance successfully.\n"
                "ACTION: run_docker\n"
                "OBSERVATION: Docker containers started: application online."
            )
            
    # -------------------------------------------------------------
    # Step 6: Write database migrations schema.sql
    # -------------------------------------------------------------
    elif call_count == 6:
        if not schema_fixed:
            return (
                "THOUGHT: Writing sql setup queries to schema.sql.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to schema.sql successfully (warning: contains invalid SQL datatype syntax)"
            )
        else:
            return (
                "THOUGHT: Re-writing schema.sql with corrected database datatypes.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to schema.sql successfully"
            )
            
    # -------------------------------------------------------------
    # Step 7: Configure server endpoints (unrelated step)
    # -------------------------------------------------------------
    elif call_count == 7:
        return (
            "THOUGHT: Mapping local network ports.\n"
            "ACTION: configure_ports\n"
            "OBSERVATION: Ports mapped: 8080 redirected to 80."
        )
        
    # -------------------------------------------------------------
    # Step 8: Execute schema migrations (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 8:
        if not schema_fixed:
            # Divergence 3: schema.sql table layout fails
            return (
                "THOUGHT: Migrating table schemas.\n"
                "ACTION: execute_migrations\n"
                "OBSERVATION: Error: sqlite3.OperationalError: table creation failed: schema.sql contains syntax error"
            )
        else:
            return (
                "THOUGHT: Migrating table schemas successfully.\n"
                "ACTION: execute_migrations\n"
                "OBSERVATION: Migration succeeded: tables created."
            )
            
    # -------------------------------------------------------------
    # Step 9: Configure load balancer properties
    # -------------------------------------------------------------
    elif call_count == 9:
        return (
            "THOUGHT: Registering microservice routes.\n"
            "ACTION: update_routing\n"
            "OBSERVATION: Routing updated successfully."
        )
        
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Full DevOps deployment completed successfully. DriftGuard resolved 3 distinct dependency failures:\n"
            "  1. Step 4 failed compiling utils.js -> Rolled back to step 2 (undid S3).\n"
            "  2. Step 5 failed running docker-compose.yml -> Rolled back to step 1 (undid S2, S3, S4).\n"
            "  3. Step 8 failed on schema.sql schema -> Rolled back to step 5 (undid S6, S7)."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=devops_deploy_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting 10-step Multi-Rollback DevOps Deploy Demo! Open http://localhost:8050 in your browser.")
result = wrapper.run(goal="Setup docker, compile frontend application, and execute database migration script.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
