"""
LangGraph Multi-Agent Workflow: Step 2 (Two Collaborative Agents)
-----------------------------------------------------------------
Flow:
  START -> [Agent 1: Researcher] -> [Agent 2: Reviewer] -> END

Concepts demonstrated:
1. Agent State Schema with multi-step fields (draft -> final_report).
2. Agent Handoff: Node 1 outputs a draft; Node 2 reads that draft and refines it.
3. Linear Sequential Edge Pipeline.
"""

from typing import TypedDict, Optional
from core.llm import call_gemini


# ============================================================================
# 1. DEFINE STATE (The shared clipboard across all agents)
# ============================================================================
class ResearchState(TypedDict):
    prompt: str
    draft: Optional[str]         # Populated by Agent 1 (Researcher)
    final_report: Optional[str]  # Populated by Agent 2 (Reviewer)
    step_count: int


# ============================================================================
# 2. DEFINE AGENT NODES
# ============================================================================
def researcher_node(state: ResearchState) -> dict:
    """
    Agent 1: Lead Researcher
    Analyzes the user's prompt and produces a detailed, fact-filled initial draft.
    """
    prompt = state.get("prompt", "")
    print("  [LangGraph Agent 1: 'researcher' gathering facts and drafting...]")

    researcher_prompt = f"""You are an Expert Research Analyst.
User Query: {prompt}

Provide a comprehensive, detailed, and fact-focused analysis answering this query.
Focus on providing rich substance, insights, and data points."""

    draft_answer = call_gemini(researcher_prompt)

    # Return update to state (populates 'draft')
    return {
        "draft": draft_answer,
        "step_count": state.get("step_count", 0) + 1
    }


def reviewer_node(state: ResearchState) -> dict:
    """
    Agent 2: Senior Editor & Reviewer
    Reads Agent 1's draft, critiques it, organizes it into structured sections,
    and produces the executive final report.
    """
    prompt = state.get("prompt", "")
    draft = state.get("draft", "")
    print("  [LangGraph Agent 2: 'reviewer' critiquing and structuring final report...]")

    reviewer_prompt = f"""You are a Senior Editor and Executive Reviewer.
Original Topic: {prompt}

Initial Draft from Researcher Agent:
\"\"\"
{draft}
\"\"\"

Your task:
1. Review and polish the researcher's draft.
2. Structure the output with:
   - 📌 Executive Summary
   - 🔍 Key Findings & Insights
   - 💡 Strategic Recommendations
3. Ensure crisp formatting, eliminate fluff, and make it executive-ready."""

    final_report = call_gemini(reviewer_prompt)

    # Return update to state (populates 'final_report')
    return {
        "final_report": final_report,
        "step_count": state.get("step_count", 0) + 1
    }


# ============================================================================
# 3. BUILD AND COMPILE THE MULTI-AGENT GRAPH
# ============================================================================
def build_research_graph():
    """Builds a two-agent sequential LangGraph workflow"""
    try:
        from langgraph.graph import StateGraph, START, END

        # Initialize graph with our shared schema
        workflow = StateGraph(ResearchState)

        # Step A: Register the two Agent Nodes
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("reviewer", reviewer_node)

        # Step B: Define the Edges (The Agent Handoff Pipeline)
        # START -> researcher
        workflow.add_edge(START, "researcher")
        
        # researcher -> reviewer (Handoff: Agent 1 passes state to Agent 2!)
        workflow.add_edge("researcher", "reviewer")
        
        # reviewer -> END (Workflow completes)
        workflow.add_edge("reviewer", END)

        # Compile into executable Runnable
        return workflow.compile()

    except ImportError:
        return None


# Cached compiled graph instance
_cached_graph = None


def get_graph():
    """Returns the compiled graph, building it once if available"""
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_research_graph()
    return _cached_graph


# ============================================================================
# 4. ENTRY FUNCTION (Invoked by main.py)
# ============================================================================
def run_research(prompt: str) -> str:
    """
    Executes the multi-agent research workflow.
    """
    graph = get_graph()

    if graph is not None:
        initial_state: ResearchState = {
            "prompt": prompt,
            "draft": None,
            "final_report": None,
            "step_count": 0
        }
        # Invoke the multi-agent graph
        result = graph.invoke(initial_state)
        
        # Return the final report produced by Agent 2 (Reviewer)
        return result.get("final_report", "⚠️ No report generated in final state.")
    else:
        print("ℹ️ Note: 'langgraph' package not detected. Running direct fallback...\n")
        return call_gemini(prompt)
