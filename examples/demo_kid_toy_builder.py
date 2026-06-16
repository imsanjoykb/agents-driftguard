import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# Simulation state variables tracking the spaceship building steps
call_count = 0
base_fixed = False
power_fixed = False
wings_fixed = False

def lego_agent_executor(prompt: str) -> str:
    global call_count, base_fixed, power_fixed, wings_fixed
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # Trace the feedback injected in prompt during rollback recovery
    if "ROLLBACK" in prompt:
        if "cockpit_base" in prompt:
            base_fixed = True
        elif "power_box" in prompt:
            power_fixed = True
        elif "wings_module" in prompt:
            wings_fixed = True

    print(f"  [LEGO Builder Call #{call_count}] (BaseFixed={base_fixed}, PowerFixed={power_fixed}, WingsFixed={wings_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Open instructions
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Opening instruction manual to page 1.\n"
            "ACTION: open_manual\n"
            "OBSERVATION: Instructions loaded."
        )
        
    # -------------------------------------------------------------
    # Step 2: Build cockpit base
    # -------------------------------------------------------------
    elif call_count == 2:
        if not base_fixed:
            return (
                "THOUGHT: Assembling cockpit frame.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to cockpit_base successfully (used thin 2x2 connector blocks instead of wide 4x4 ones)"
            )
        else:
            return (
                "THOUGHT: Assembling cockpit frame using the correct wide 4x4 connector blocks.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to cockpit_base successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Sort bricks by color (unrelated)
    # -------------------------------------------------------------
    elif call_count == 3:
        return (
            "THOUGHT: Sorting red and blue blocks.\n"
            "ACTION: sort_blocks\n"
            "OBSERVATION: Blocks sorted into groups."
        )
        
    # -------------------------------------------------------------
    # Step 4: Assemble landing gear (Fails if base wrong)
    # -------------------------------------------------------------
    elif call_count == 4:
        if not base_fixed:
            # Divergence 1: Landing gear falls off
            return (
                "THOUGHT: Fitting landing gear struts under cockpit base.\n"
                "ACTION: attach_landing_gear\n"
                "OBSERVATION: Error: The landing gear struts fell off! The connector pegs under cockpit_base are too thin"
            )
        else:
            return (
                "THOUGHT: Fitting landing gear struts under cockpit base.\n"
                "ACTION: attach_landing_gear\n"
                "OBSERVATION: Landing gear snapped securely in place."
            )
            
    # -------------------------------------------------------------
    # Step 5: Install battery module
    # -------------------------------------------------------------
    elif call_count == 5:
        if not power_fixed:
            return (
                "THOUGHT: Placing the engine power block.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to power_box successfully (batteries put in backwards, reversed polarity)"
            )
        else:
            return (
                "THOUGHT: Placing engine power block with correct battery direction.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to power_box successfully"
            )
            
    # -------------------------------------------------------------
    # Step 6: Assemble wings
    # -------------------------------------------------------------
    elif call_count == 6:
        if not wings_fixed:
            return (
                "THOUGHT: Constructing wing components.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to wings_module successfully (wings assembled pointing backward instead of forward)"
            )
        else:
            return (
                "THOUGHT: Constructing wing components pointing forward.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to wings_module successfully"
            )
            
    # -------------------------------------------------------------
    # Step 7: Turn on LED wing lights (Fails if batteries wrong)
    # -------------------------------------------------------------
    elif call_count == 7:
        if not power_fixed:
            # Divergence 2: LED lights do not glow
            return (
                "THOUGHT: Connecting light cables to battery.\n"
                "ACTION: test_lights\n"
                "OBSERVATION: Error: The LED lights do not glow! The batteries in power_box are reversed"
            )
        else:
            return (
                "THOUGHT: Connecting light cables successfully.\n"
                "ACTION: test_lights\n"
                "OBSERVATION: LED lights glow green and blue successfully."
            )
            
    # -------------------------------------------------------------
    # Step 8: Attach wings to cockpit (Fails if backward)
    # -------------------------------------------------------------
    elif call_count == 8:
        if not wings_fixed:
            # Divergence 3: Wings don't align
            return (
                "THOUGHT: Attaching wings to cockpit_base.\n"
                "ACTION: attach_wings\n"
                "OBSERVATION: Error: The wings do not fit! The connectors on wings_module are reversed"
            )
        else:
            return (
                "THOUGHT: Attaching wings to cockpit_base successfully.\n"
                "ACTION: attach_wings\n"
                "OBSERVATION: Wings clicked on successfully: spaceship complete."
            )
            
    # -------------------------------------------------------------
    # Step 9: Place LEGO astronaut pilot in cockpit
    # -------------------------------------------------------------
    elif call_count == 9:
        return (
            "THOUGHT: Seating the astronaut figure in the control seat.\n"
            "ACTION: seat_pilot\n"
            "OBSERVATION: Astronaut pilot is seated."
        )
        
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Amazing! The LEGO Spaceship is complete, lights up, and is ready for flight! "
            "DriftGuard successfully resolved 3 distinct building issues:\n"
            "  1. Step 4 failed (landing gear) -> Rolled back to step 1 to build wider cockpit_base pegs (undid S2).\n"
            "  2. Step 7 failed (no lights) -> Rolled back to step 4 to correct power_box battery polarity (undid S5).\n"
            "  3. Step 8 failed (wings misaligned) -> Rolled back to step 5 to build wings_module pointing forward (undid S6)."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=lego_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting Elaborate LEGO Spaceship Builder demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Build a complete LEGO spaceship model.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
