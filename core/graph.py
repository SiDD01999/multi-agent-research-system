"""
LangGraph Multi-Agent Workflow: Step 5 (MemorySaver & Conversational State)
-------------------------------------------------------------------------
Pattern: Evaluator-Optimizer with Checkpointer Memory & State Reducers

Architecture:
  START ──► [Agent 1: Researcher] ──► [Agent 2: LLM Judge] ──► should_continue?
                 ▲                                                    │
                 │                    (Score < 8.5 & Improving)       │
                 └─────────────────── "revise" ◄──────────────────────┤
                                                                      │
                                      (Score >= 8.5 / Plateau / Limit)│
                                      "finalize" ─────────────────────► [Formatter] ──► END

Key Concepts:
1. MemorySaver Checkpointer: Persists state snapshots across multiple turns via thread_id.
2. State Reducers (Annotated[List[dict], operator.add]): Automatically appends messages across turns.
3. Multi-Turn Conversational Awareness: Researcher references previous conversation context.
4. Multi-dimensional scoring rubric & Plateau detection.
"""

import json
import operator
from typing import TypedDict, Optional, List, Annotated
from core.llm import call_gemini
from core.logger import ExecutionLogger, get_current_logger, set_current_logger

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
    # 🆕 State Reducer: Appends new turns to chat history instead of overwriting!
    chat_history: Annotated[List[dict], operator.add]
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
    - Conversational Awareness: Inspects previous turns in chat_history.
    - Round 0: Gathers comprehensive facts for the current prompt.
    - Round 1+: Reads specific gaps identified by the LLM Judge to fix weaknesses.
    """
    prompt = state.get("prompt", "")
    evaluation = state.get("evaluation")
    revision_count = state.get("revision_count", 0)
    chat_history = state.get("chat_history", [])
    logger = get_current_logger()

    # Format previous turns context (taking up to last 4 messages for concise context)
    history_context = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-4:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate older lengthy assistant answers in context
            preview = content[:250] + "..." if len(content) > 250 else content
            history_lines.append(f"- {role}: {preview}")
        history_context = "Previous Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    if evaluation and evaluation.get("gaps"):
        gaps_list = "\n- ".join(evaluation["gaps"])
        print(f"\n  [LangGraph Agent 1: 'researcher' targeting specific gaps (Revision {revision_count})...]")
        researcher_prompt = f"""You are an Expert Research Analyst.
{history_context}Original Topic: {prompt}

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
        if history_context:
            print(f"\n  [LangGraph Agent 1: 'researcher' reviewing conversation history & drafting (Revision {revision_count})...]")
        else:
            print(f"\n  [LangGraph Agent 1: 'researcher' gathering initial research (Revision {revision_count})...]")

        researcher_prompt = f"""You are an Expert Research Analyst.
{history_context}Current User Query: {prompt}

Provide an exhaustive, detailed, and fact-focused research draft answering this query.
If the query references past topics or names from the conversation history, seamlessly integrate that context.
Focus on providing rich technical substance, real-world examples, and actionable depth."""

    draft = call_gemini(researcher_prompt)

    if logger:
        logger.log_step(
            f"Agent 1: Researcher (Revision {revision_count})",
            f"[PROMPT SENT TO LLM]:\n{researcher_prompt}\n\n[DRAFT RETURNED]:\n{draft}"
        )

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
    logger = get_current_logger()

    print("  [LangGraph Agent 2: 'judge' evaluating draft against rubric...]")

    judge_prompt = f"""You are an exacting, critical LLM Judge. Evaluate the following research draft.

User Query:
\"{prompt}\"

Research Draft:
\"\"\"
{draft}
\"\"\"

Evaluation Rubric:
1. Completeness (1-10): Did it thoroughly address the user's intent?
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

    if logger:
        logger.log_scorecard(comp, depth, struct, overall, TARGET_SCORE, gaps)
        logger.log_step("Agent 2: LLM Judge Raw Output", f"[RAW JSON EVALUATION]:\n{raw_eval}")

    return {
        "evaluation": evaluation,
        "score_history": score_history
    }


def formatter_node(state: ResearchState) -> dict:
    """
    Agent 3: Executive Formatter
    Transforms the final accepted research into a crisp, polished deliverable.
    Appends the completed turn (user prompt + final report) to chat_history via reducer.
    """
    prompt = state.get("prompt", "")
    draft = state.get("draft", "")
    score = state.get("evaluation", {}).get("overall_score", 0.0)
    count = state.get("revision_count", 0)
    history = state.get("score_history", [])
    logger = get_current_logger()

    if score >= TARGET_SCORE:
        reason = f"Target Quality Met: {score}/10"
    elif count >= MAX_REVISIONS:
        reason = f"Max Revisions Reached: {count}/{MAX_REVISIONS}"
    elif len(history) >= 2 and (history[-1] - history[-2]) < PLATEAU_THRESHOLD:
        reason = f"Quality Plateau Detected"
    else:
        reason = "Completed"

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

    if logger:
        logger.log_step(f"Agent 3: Formatter ({reason})", final_report)

    return {
        "final_report": final_report,
        "exit_reason": reason,
        # 🆕 Append to chat_history via operator.add reducer!
        "chat_history": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_report}
        ]
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
    logger = get_current_logger()

    decision = "revise"
    reason = f"Score ({score}/10) < {TARGET_SCORE}. Looping back to Researcher."

    if score >= TARGET_SCORE:
        decision = "finalize"
        reason = f"Target Quality Met ({score}/10 >= {TARGET_SCORE})"
        print(f"  [Routing Decision ✅]: Target achieved ({score}/10)! Routing to Formatter.")

    elif count >= MAX_REVISIONS:
        decision = "finalize"
        reason = f"Max Revisions Reached ({count}/{MAX_REVISIONS})"
        print(f"  [Routing Decision ⚠️]: Max revisions ({count}) reached. Routing to Formatter.")

    elif len(history) >= 2 and (history[-1] - history[-2]) < PLATEAU_THRESHOLD:
        score_delta = history[-1] - history[-2]
        decision = "finalize"
        reason = f"Quality Plateau Detected (Score delta +{score_delta:.1f} < {PLATEAU_THRESHOLD})"
        print(f"  [Routing Decision 🛑]: Score plateaued (+{score_delta:.1f}). Stopping early to save tokens.")

    else:
        print(f"  [Routing Decision 🔁]: Score ({score}/10) < {TARGET_SCORE}. Looping back to Researcher.")

    if logger:
        logger.log_step(f"Routing Decision -> '{decision}'", f"Reason: {reason}")

    return decision


# ============================================================================
# 4. BUILD AND COMPILE GRAPH WITH MEMORYSAVER
# ============================================================================
_cached_checkpointer = None


def build_research_graph():
    """Builds the Evaluator-Optimizer graph with MemorySaver Checkpointer"""
    global _cached_checkpointer
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver

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

        # 🆕 Initialize MemorySaver checkpointer
        if _cached_checkpointer is None:
            _cached_checkpointer = MemorySaver()

        # Compile with checkpointer enabled!
        return workflow.compile(checkpointer=_cached_checkpointer)

    except ImportError:
        return None


# Cached compiled graph instance
_cached_graph = None


def get_graph():
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_research_graph()
    return _cached_graph


# Fallback memory for non-langgraph execution
_fallback_chat_history: List[dict] = []


# ============================================================================
# 5. ENTRY FUNCTION (Invoked by main.py)
# ============================================================================
def run_research(prompt: str, thread_id: str = "default_cli_session") -> str:
    """
    Executes the research workflow via LangGraph with MemorySaver checkpointer.
    Preserves and retrieves conversational history across turns for the thread_id.
    """
    global _fallback_chat_history
    logger = get_current_logger()
    if logger is None:
        logger = SessionLogger()
        set_current_logger(logger)

    logger.start_turn(prompt)

    graph = get_graph()

    if graph is not None:
        # Pass thread_id in config so MemorySaver identifies this conversation thread
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "prompt": prompt,
            "draft": None,
            "evaluation": None,
            "score_history": [],
            "final_report": None,
            "revision_count": 0,
            "exit_reason": None,
            # We pass empty list here so the operator.add reducer accumulates from checkpoint
            "chat_history": []
        }

        result = graph.invoke(initial_state, config=config)

        final_score = result.get("evaluation", {}).get("overall_score", 0.0) if result.get("evaluation") else 0.0
        reason = result.get("exit_reason", "Completed")
        revisions = result.get("revision_count", 0)

        logger.end_turn(reason, final_score, revisions)

        return result.get("final_report", "⚠️ No report generated in final state.")
    else:
        print("ℹ️ Note: 'langgraph' package not detected. Running direct fallback...\n")
        # Fallback with simple memory
        hist_str = ""
        if _fallback_chat_history:
            hist_str = "Conversation history:\n" + "\n".join(f"{m['role']}: {m['content'][:200]}" for m in _fallback_chat_history[-3:]) + "\n\n"
        answer = call_gemini(f"{hist_str}User: {prompt}")
        _fallback_chat_history.append({"role": "user", "content": prompt})
        _fallback_chat_history.append({"role": "assistant", "content": answer})
        logger.log_step("Direct Fallback Output", answer)
        logger.end_turn("Direct Fallback", 0.0, 0)
        return answer
