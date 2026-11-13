"""
LangGraph Multi-Agent Workflow: Step 4 (LLM-as-a-Judge with Rubric Scoring)
--------------------------------------------------------------------------
Pattern: Evaluator-Optimizer with Numerical Rubric & Plateau Detection

Architecture:
  START ──► [Agent 1: Researcher] ──► [Agent 2: LLM Judge] ──► should_continue?
                 ▲                                                    │
                 │                    (Score < 8.5 & Improving)       │
                 └─────────────────── "revise" ◄──────────────────────┤
                                                                      │
                                      (Score >= 8.5 / Plateau / Limit)│
                                      "finalize" ─────────────────────► [Formatter] ──► END

Key Concepts:
1. Multi-dimensional scoring rubric (Completeness, Depth, Structure).
2. JSON structured evaluation contract.
3. Plateau detection (halts loop if score improvement stalls: delta < 0.2).
4. Circuit breaker (MAX_REVISIONS limit).
"""

import json
import re
from typing import TypedDict, Optional, List
from core.llm import call_gemini

# ============================================================================
# CONFIGURABLE EVALUATION THRESHOLDS
# ============================================================================
TARGET_SCORE = 8.5       # Score out of 10 required to pass immediately
MAX_REVISIONS = 3        # Maximum iterations to prevent runaway loops
PLATEAU_THRESHOLD = 0.2  # Minimum score improvement needed between consecutive rounds


# ============================================================================
# 1. DEFINE STATE (The shared clipboard across all agents)
# ============================================================================
class Evaluation(TypedDict):
    completeness: int
    technical_depth: int
    structure: int
    overall_score: float
    gaps: List[str]


class ResearchState(TypedDict):
    prompt: str
    draft: Optional[str]
    evaluation: Optional[Evaluation]
    score_history: List[float]
    final_report: Optional[str]
    revision_count: int
    exit_reason: Optional[str]


# ============================================================================
# HELPER: Robust JSON Extraction
# ============================================================================
def extract_json(raw_text: str) -> dict:
    """Extract and parse JSON object from LLM response reliably"""
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback heuristic if json fails
        return {
            "completeness": 7,
            "technical_depth": 7,
            "structure": 7,
            "gaps": ["Expand technical implementation details and provide concrete data points."]
        }


# ============================================================================
# 2. DEFINE AGENT NODES
# ============================================================================
def researcher_node(state: ResearchState) -> dict:
    """
    Agent 1: Researcher / Generator
    - Round 0: Gathers comprehensive facts for the user prompt.
    - Round 1+: Reads the specific gaps identified by the LLM Judge to fix weaknesses.
    """
    prompt = state.get("prompt", "")
    evaluation = state.get("evaluation")
    revision_count = state.get("revision_count", 0)

    if evaluation and evaluation.get("gaps"):
        gaps_list = "\n- ".join(evaluation["gaps"])
        print(f"\n  [LangGraph Agent 1: 'researcher' targeting specific gaps (Revision {revision_count})...]")
        researcher_prompt = f"""You are an Expert Research Analyst.
Original Topic: {prompt}

Previous Draft:
\"\"\"
{state.get('draft', '')}
\"\"\"

The Senior LLM Judge deducted points for the following specific gaps:
- {gaps_list}

Your task:
Rewrite and upgrade the research to thoroughly resolve these gaps.
Add deep technical substance, concrete numbers/examples, and eliminate any remaining weaknesses."""
    else:
        print(f"\n  [LangGraph Agent 1: 'researcher' gathering initial research (Revision {revision_count})...]")
        researcher_prompt = f"""You are an Expert Research Analyst.
User Query: {prompt}

Provide an exhaustive, detailed, and fact-focused research draft answering this query.
Focus on providing rich technical substance, real-world examples, and actionable depth."""

    draft = call_gemini(researcher_prompt)

    return {
        "draft": draft,
        "revision_count": revision_count + 1
    }


def judge_node(state: ResearchState) -> dict:
    """
    Agent 2: LLM-as-a-Judge (The Evaluator)
    Scores the draft against a strict 3-dimensional rubric:
    1. Completeness (1-10)
    2. Technical Depth (1-10)
    3. Structure & Actionability (1-10)
    Returns structured JSON with sub-scores, overall score, and specific gaps.
    """
    prompt = state.get("prompt", "")
    draft = state.get("draft", "")
    score_history = list(state.get("score_history", []))

    print("  [LangGraph Agent 2: 'judge' evaluating draft against rubric...]")

    judge_prompt = f"""You are an exacting, critical LLM Judge. Evaluate the following research draft.

Original Query:
\"{prompt}\"

Research Draft:
\"\"\"
{draft}
\"\"\"

Evaluation Rubric:
1. Completeness (1-10): Did it answer all parts of the user query?
2. Technical Depth (1-10): Does it include concrete architectural details, numbers, trade-offs, and examples rather than generic advice?
3. Structure & Actionability (1-10): Is it logically structured with clear takeaways?

Be strict and demanding. A draft with generic platitudes or missing edge cases should not score above 7.

Return ONLY a valid JSON object matching this exact schema:
{{
  "completeness": <int 1-10>,
  "technical_depth": <int 1-10>,
  "structure": <int 1-10>,
  "gaps": [
    "<specific actionable missing detail 1>",
    "<specific actionable missing detail 2>"
  ]
}}"""

    raw_eval = call_gemini(judge_prompt)
    data = extract_json(raw_eval)

    comp = int(data.get("completeness", 7))
    depth = int(data.get("technical_depth", 7))
    struct = int(data.get("structure", 7))
    overall = round((comp + depth + struct) / 3.0, 1)
    gaps = data.get("gaps", [])

    evaluation: Evaluation = {
        "completeness": comp,
        "technical_depth": depth,
        "structure": struct,
        "overall_score": overall,
        "gaps": gaps
    }

    score_history.append(overall)

    # Print live formatted scorecard in terminal
    print(f"  ┌── 📊 LLM Judge Scorecard ──────────────────────┐")
    print(f"  │  • Completeness:     {comp}/10")
    print(f"  │  • Technical Depth:  {depth}/10")
    print(f"  │  • Structure:        {struct}/10")
    print(f"  │  ⭐ Overall Score:    {overall}/10 (Target: {TARGET_SCORE}/10)")
    if gaps:
        print(f"  │  ⚠️  Key Gap: {gaps[0][:48]}...")
    print(f"  └────────────────────────────────────────────────┘")

    return {
        "evaluation": evaluation,
        "score_history": score_history
    }


def formatter_node(state: ResearchState) -> dict:
    """
    Agent 3: Executive Formatter
    Transforms the final accepted research into a crisp, polished deliverable.
    """
    prompt = state.get("prompt", "")
    draft = state.get("draft", "")
    score = state.get("evaluation", {}).get("overall_score", 0.0)
    reason = state.get("exit_reason", "Completed")

    print(f"\n  [LangGraph Formatter: Producing final deliverable ({reason})...]")

    formatter_prompt = f"""You are an Executive Editor.
Original Topic: {prompt}
Final Verified Research:
\"\"\"
{draft}
\"\"\"

Produce an executive-ready final report with:
- 📌 Executive Summary
- 🔍 Technical Architecture & Key Findings
- 💡 Strategic Recommendations & Trade-offs
Include a small footer badge noting the Verified Quality Score ({score}/10)."""

    final_report = call_gemini(formatter_prompt)

    return {
        "final_report": final_report
    }


# ============================================================================
# 3. CONDITIONAL ROUTING FUNCTION
# ============================================================================
def should_continue(state: ResearchState) -> str:
    """
    Deterministic Routing Decision:
    1. Score >= TARGET_SCORE (8.5) -> finalize (Passed!)
    2. Revisions >= MAX_REVISIONS -> finalize (Hit limit)
    3. Score Plateau (Delta < 0.2 between consecutive rounds) -> finalize (Diminishing returns)
    4. Otherwise -> revise (Loop back to Researcher)
    """
    eval_data = state.get("evaluation", {})
    score = eval_data.get("overall_score", 0.0)
    history = state.get("score_history", [])
    count = state.get("revision_count", 0)

    # Condition 1: Target quality achieved!
    if score >= TARGET_SCORE:
        state["exit_reason"] = f"Target Quality Met ({score}/10 >= {TARGET_SCORE})"
        print(f"  [Routing Decision ✅]: Target achieved ({score}/10)! Routing to Formatter.")
        return "finalize"

    # Condition 2: Max revisions reached
    if count >= MAX_REVISIONS:
        state["exit_reason"] = f"Max Revisions Reached ({count}/{MAX_REVISIONS})"
        print(f"  [Routing Decision ⚠️]: Max revisions ({count}) reached. Routing to Formatter.")
        return "finalize"

    # Condition 3: Plateau detected (no meaningful improvement over last round)
    if len(history) >= 2:
        score_delta = history[-1] - history[-2]
        if score_delta < PLATEAU_THRESHOLD:
            state["exit_reason"] = f"Quality Plateau Detected (Score delta +{score_delta:.1f} < {PLATEAU_THRESHOLD})"
            print(f"  [Routing Decision 🛑]: Score plateaued (+{score_delta:.1f}). Stopping early to save tokens.")
            return "finalize"

    # Condition 4: Below threshold and still improving -> Loop back
    print(f"  [Routing Decision 🔁]: Score ({score}/10) < {TARGET_SCORE}. Looping back to Researcher.")
    return "revise"


# ============================================================================
# 4. BUILD AND COMPILE THE GRAPH
# ============================================================================
def build_research_graph():
    """Builds the Evaluator-Optimizer graph with LLM Judge & Plateau Detection"""
    try:
        from langgraph.graph import StateGraph, START, END

        workflow = StateGraph(ResearchState)

        # 1. Register Nodes
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("judge", judge_node)
        workflow.add_node("formatter", formatter_node)

        # 2. Add Fixed Edges
        workflow.add_edge(START, "researcher")
        workflow.add_edge("researcher", "judge")
        workflow.add_edge("formatter", END)

        # 3. Add Conditional Edge from Judge
        workflow.add_conditional_edges(
            "judge",
            should_continue,
            {
                "revise": "researcher",  # 🔁 Loop back with gaps
                "finalize": "formatter"  # 🏁 Finalize into executive brief
            }
        )

        return workflow.compile()

    except ImportError:
        return None


# Cached compiled graph instance
_cached_graph = None


def get_graph():
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_research_graph()
    return _cached_graph


# ============================================================================
# 5. ENTRY FUNCTION (Invoked by main.py)
# ============================================================================
def run_research(prompt: str) -> str:
    """
    Executes the research workflow via LangGraph with LLM-as-a-Judge.
    """
    graph = get_graph()

    if graph is not None:
        initial_state: ResearchState = {
            "prompt": prompt,
            "draft": None,
            "evaluation": None,
            "score_history": [],
            "final_report": None,
            "revision_count": 0,
            "exit_reason": None
        }

        result = graph.invoke(initial_state)

        return result.get("final_report", "⚠️ No report generated in final state.")
    else:
        print("ℹ️ Note: 'langgraph' package not detected. Running direct fallback...\n")
        return call_gemini(prompt)
