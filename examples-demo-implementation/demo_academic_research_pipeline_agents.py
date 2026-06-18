import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# Simulation state variables tracking the academic research fixes
call_count = 0
bibtex_fixed = False
script_fixed = False

def academic_research_executor(prompt: str) -> str:
    global call_count, bibtex_fixed, script_fixed
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visualization
    
    # Trace the feedback injected in prompt during rollback recovery
    if "ROLLBACK" in prompt:
        if "citations.bib" in prompt:
            bibtex_fixed = True
        elif "analyze.py" in prompt:
            script_fixed = True

    print(f"  [Agent Executing Call #{call_count}] (BibTeXFixed={bibtex_fixed}, PythonScriptFixed={script_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Query arXiv literature
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Searching academic databases for agent reliability research papers.\n"
            "ACTION: query_arxiv\n"
            "OBSERVATION: Found 3 primary reference papers."
        )
        
    # -------------------------------------------------------------
    # Step 2: Write bibliography citations database
    # -------------------------------------------------------------
    elif call_count == 2:
        if not bibtex_fixed:
            return (
                "THOUGHT: Saving citation records into citations.bib file.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to citations.bib successfully (warning: missing closing brace in BibTeX entry)"
            )
        else:
            return (
                "THOUGHT: Re-writing citations.bib with correct BibTeX block braces syntax.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to citations.bib successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Fetch lab dataset values
    # -------------------------------------------------------------
    elif call_count == 3:
        return (
            "THOUGHT: Loading raw experimental dataset properties.\n"
            "ACTION: fetch_dataset\n"
            "OBSERVATION: Dataset parameters loaded: 1000 sample rows."
        )
        
    # -------------------------------------------------------------
    # Step 4: Write python data-plotting script analyze.py
    # -------------------------------------------------------------
    elif call_count == 4:
        if not script_fixed:
            return (
                "THOUGHT: Generating analysis script analyze.py to compute and plot results.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to analyze.py successfully (warning: using uninstalled library name)"
            )
        else:
            return (
                "THOUGHT: Re-writing analyze.py utilizing correct standard python libraries.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to analyze.py successfully"
            )
            
    # -------------------------------------------------------------
    # Step 5: Execute analyze.py to plot figures (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 5:
        if not script_fixed:
            # Divergence 1: Python script import error
            return (
                "THOUGHT: Running data plotting analysis.\n"
                "ACTION: execute_script\n"
                "OBSERVATION: Error compiling build: ModuleNotFoundError: No module named 'analyze.py'"
            )
        else:
            return (
                "THOUGHT: Running data analysis script successfully.\n"
                "ACTION: execute_script\n"
                "OBSERVATION: Data processed. saved to plot.png successfully"
            )
            
    # -------------------------------------------------------------
    # Step 6: Download university LaTeX template files (unrelated)
    # -------------------------------------------------------------
    elif call_count == 6:
        return (
            "THOUGHT: Fetching paper layout style sheets.\n"
            "ACTION: fetch_template\n"
            "OBSERVATION: Style files retrieved."
        )
        
    # -------------------------------------------------------------
    # Step 7: Write LaTeX paper manuscript layout
    # -------------------------------------------------------------
    elif call_count == 7:
        return (
            "THOUGHT: Formatting LaTeX structure drafting paper draft.\n"
            "ACTION: write_file\n"
            "OBSERVATION: saved to manuscript.tex successfully (linked to citations.bib and plot.png)"
        )
        
    # -------------------------------------------------------------
    # Step 8: Query GPU cluster nodes statistics (unrelated)
    # -------------------------------------------------------------
    elif call_count == 8:
        return (
            "THOUGHT: Checking GPU cluster parameters.\n"
            "ACTION: check_nodes\n"
            "OBSERVATION: GPU nodes online."
        )
        
    # -------------------------------------------------------------
    # Step 9: Compile LaTeX PDF paper (Fails if not fixed)
    # -------------------------------------------------------------
    elif call_count == 9:
        if not bibtex_fixed:
            # Divergence 2: LaTeX/BibTeX syntax build failure
            return (
                "THOUGHT: Running pdflatex compilation.\n"
                "ACTION: compile_latex\n"
                "OBSERVATION: Error compiling build: BibTeX Error: unexpected EOF in citations.bib"
            )
        else:
            return (
                "THOUGHT: Compiling LaTeX paper successfully.\n"
                "ACTION: compile_latex\n"
                "OBSERVATION: pdflatex compiled. saved to paper.pdf successfully"
            )
            
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Academic Research Paper pipeline completed successfully. DriftGuard resolved 2 research file dependency issues:\n"
            "  1. Step 5 failed on python plotting script analyze.py -> DriftGuard rolled back directly to step 3 (undid S4).\n"
            "  2. Step 9 failed on BibTeX syntax in citations.bib -> DriftGuard rolled back directly to step 1 (undid S2 to S8), "
            "allowing the agent to correct the citation structure and compile paper.pdf successfully."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=academic_research_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting Academic Research paper pipeline demo! Open http://localhost:8050 in your browser.")
result = wrapper.run(goal="Analyze research dataset, compile bibliography files, and build paper PDF draft.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
