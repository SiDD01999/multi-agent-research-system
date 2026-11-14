---
name: langgraph-multi-agent-system
description: >-
  Comprehensive architectural reference, implementation patterns, runbooks, and guidelines
  for the multi-agent research system built with LangGraph, Google Gemini, LLM-as-a-Judge
  rubric scoring, MemorySaver session memory, and session execution logging.
  Activate this skill when continuing, maintaining, extending, or building upon this project
  in future sessions.
---

# 🤖 Multi-Agent Research System (LangGraph + Gemini)

This skill serves as the persistent memory, architectural blueprint, and development guide for the **Multi-Agent Research System**. Use this runbook to immediately resume development, understand past design choices, and build next-phase features without losing context.

---

## 📁 1. Project Directory & Architecture Contract

The system enforces strict decoupling: **`main.py` is purely the interface runner and is never modified when adding or altering agent logic**.

```
multi-agent-research-system/
├── .env                  # Private API key: GEMINI_API_KEY
├── requirements.txt      # langgraph, langchain-google-genai, requests, python-dotenv
├── main.py               # Decoupled CLI session interface with MemorySaver thread_id
├── logs/                 # Ignored by git; session files: execution_logs_<timestamp>.txt
├── core/
│   ├── __init__.py       # Package marker
│   ├── llm.py            # Standard Gemini API caller (urllib, zero external client deps)
│   ├── logger.py         # SessionLogger: records entire interactive session until Ctrl+C
│   └── graph.py          # State schema, 3 agent nodes, conditional edge, MemorySaver checkpointer
├── .agents/skills/       # Project skills directory
│   └── langgraph-multi-agent-system/SKILL.md  # This skill specification
└── README.md             # High-level documentation & interview cheat sheet
```

---

## 🧠 2. Core Architectural Pillars Implemented

### Pillar 1: The Evaluator-Optimizer Cyclic Graph
```
START ──► [Memory Compactor] ──► [Agent 1: Researcher] ──► [Agent 2: LLM Judge] ──► should_continue?
                                       ▲                                                    │
                                       │                    (Score < 8.5 & Improving)       │
                                       └─────────────────── "revise" ◄──────────────────────┤
                                                                                            │
                                                            (Score >= 8.5 / Plateau / Limit)│
                                                            "finalize" ─────────────────────► [Agent 3: Formatter] ──► END
```

### Pillar 2: State Schema & Smart Reducer (`core/graph.py`)
```python
from typing import TypedDict, Optional, List, Annotated, Any

def manage_chat_history(existing: Optional[List[dict]], incoming: Any) -> List[dict]:
    """Smart reducer supporting both message appending and compactor replacement."""
    existing = list(existing or [])
    if not incoming:
        return existing
    if isinstance(incoming, dict) and incoming.get("mode") == "replace":
        return list(incoming.get("messages", []))
    if isinstance(incoming, list):
        return existing + incoming
    return existing

class ResearchState(TypedDict):
    prompt: str
    summary: Optional[str]                             # 🆕 Progressive rolling summary
    chat_history: Annotated[List[dict], manage_chat_history]  # 🆕 Bounded active buffer
    draft: Optional[str]
    evaluation: Optional[Evaluation]
    score_history: List[float]       # Tracks progression across cycles
    final_report: Optional[str]
    revision_count: int              # Tracks cycles per turn
    exit_reason: Optional[str]
```

### Pillar 3: LLM-as-a-Judge with Quantitative Rubric & Plateau Detection
* **Target Quality Gate**: `TARGET_SCORE = 8.5` (out of 10).
* **Hard Circuit Breaker**: `MAX_REVISIONS = 3` (prevents infinite runaway loops).
* **Plateau Detection**: If $\Delta \text{score} < 0.2$ between consecutive cycles, halts early to eliminate diminishing returns and save API tokens.
* **Targeted Gap Correction**: Researcher receives specific items from `evaluation["gaps"]` rather than blindly regenerating.

### Pillar 4: Progressive Summary-Buffer Memory (`MemorySaver` + Compactor)
* Compiled with `checkpointer = MemorySaver()`.
* Invoked with `config = {"configurable": {"thread_id": f"session_{timestamp}"}}`.
* **Memory Compactor Pre-Node**: Automatically runs before Researcher. When active messages exceed `BUFFER_LIMIT = 4`, it compresses older turns into a rolling `summary` and keeps only the latest turn in `chat_history`.
* **Researcher Context**: Combines both the persistent executive summary (long-term) and the immediate dialogue buffer (short-term). Prevents unbounded checkpoint bloat and token waste.

### Pillar 5: Continuous Session Execution Logger (`core/logger.py`)
* Created once per `python main.py` invocation: `logs/execution_logs_<timestamp>.txt`.
* Appends `TURN #1`, `TURN #2`, `TURN #3`... sequentially.
* Closed gracefully on `Ctrl+C` or `exit` with session duration and total turns.

---

## 🛠️ 3. How to Run & Verify

1. **Start the interactive CLI**:
   ```bash
   python main.py
   ```
2. **Verify active components in terminal**:
   - `🔑 API Key configured.`
   - `📁 [Session Logger]: Recording session to logs/execution_logs_<timestamp>.txt`
   - `🧠 [MemorySaver]: Thread ID active: 'session_<timestamp>'`
3. **Test single-turn prompt**:
   - Type prompt $\rightarrow$ Watch Researcher $\rightarrow$ Watch Judge scorecard $\rightarrow$ Watch Formatter.
4. **Test multi-turn memory**:
   - Ask follow-up: *"You remember what we discussed? Now compare it with..."*
   - Verify Researcher prints: `[LangGraph Agent 1: 'researcher' reviewing conversation history & drafting...]`.

---

## 🔮 4. Next Planned Development Milestones

When continuing in a new session, choose from these planned advancements:

### Milestone A: Google Search Tool Grounding (Real-Time Web Research)
* **Goal**: Enable Agent 1 to query live web search when answering questions requiring current facts, pricing, or news.
* **Pattern**: Bind `tools = [{'type': 'google_search'}]` via Google GenAI SDK or LangChain ToolNode.
* **Architecture**:
  ```
  START ──► researcher ──► tool_node (Google Search) ──► researcher ──► judge ...
  ```

### Milestone B: Human-in-the-Loop (Approval Interrupts)
* **Goal**: Pause execution before the Formatter publishes the final report, allowing the user to approve or inject custom guidance.
* **Pattern**: `workflow.compile(checkpointer=checkpointer, interrupt_before=["formatter"])`.

### Milestone C: Persistent Disk Checkpointer (SQLite / PostgreSQL)
* **Goal**: Persist memory across terminal restarts so sessions survive after exiting `main.py`.
* **Pattern**: Replace `MemorySaver` with `SqliteSaver.from_conn_string("research_history.db")`.

---

## 🎯 5. Job Interview Concepts Mastered in this Project

1. **State Reducers (`operator.add`)**: Prevents list overwriting; accumulates messages across turns.
2. **Cyclic Graphs**: Native cycle support in LangGraph vs. linear DAG constraints in traditional pipelines.
3. **Evaluator-Optimizer Pattern**: Structured Pydantic/JSON rubrics over subjective evaluations.
4. **Moving Goalpost Mitigation**: Using numerical thresholds and plateau detection to guarantee termination.
5. **Decoupled Agent Engineering**: Separating interface (`main.py`), execution runtime (`core/graph.py`), and tool/client layers (`core/llm.py`).
