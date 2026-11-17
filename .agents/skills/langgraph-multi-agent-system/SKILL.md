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
│   ├── llm.py            # Gemini API caller + Google Search Grounding (urllib, zero external client deps)
│   ├── tools.py          # Real-time web search tool (Google Search Grounding + DuckDuckGo fallback)
│   ├── logger.py         # SessionLogger: records entire interactive session until Ctrl+C
│   └── graph.py          # State schema, 4 agent nodes, conditional edge, MemorySaver checkpointer
├── .agents/skills/       # Project skills directory
│   └── langgraph-multi-agent-system/SKILL.md  # This skill specification
└── README.md             # High-level documentation & interview cheat sheet
```

---

## 🧠 2. Core Architectural Pillars Implemented

### Pillar 1: The Evaluator-Optimizer Cyclic Graph with Web Search
```
START ──► [Memory Compactor] ──► [Web Search Agent] ──► [Agent 1: Researcher] ──► [Agent 2: LLM Judge] ──► should_continue?
                                         ▲                                                                         │
                                         │                           (Score < 8.5 & Improving)                     │
                                         └────────────────────────── "revise" ◄────────────────────────────────────┤
                                                                                                                   │
                                                                           (Score >= 8.5 / Plateau / Limit)        │
                                                                           "finalize" ─────────────────────────────► [Formatter] ──► END
```

### Pillar 2: State Schema & Smart Reducer (`core/graph.py`)
```python
from typing import TypedDict, Optional, List, Annotated, Any

class Source(TypedDict):
    title: str
    url: str

class Evaluation(TypedDict):
    completeness: int        # 1-10
    technical_depth: int     # 1-10
    structure: int           # 1-10
    grounding: int           # 1-10 🆕 Real-world web sources, facts, and citations
    overall_score: float     # Average of 4 dimensions
    gaps: List[str]          # Actionable deficiencies

class ResearchState(TypedDict):
    prompt: str
    summary: Optional[str]                             # Progressive rolling summary
    chat_history: Annotated[List[dict], manage_chat_history]  # Bounded active buffer
    draft: Optional[str]
    search_queries: List[str]                          # 🆕 Search queries executed
    sources: List[Source]                              # 🆕 Clickable web source citations
    web_evidence: Optional[str]                        # 🆕 Live extracted evidence
    evaluation: Optional[Evaluation]
    score_history: List[float]       # Tracks progression across cycles
    final_report: Optional[str]
    revision_count: int              # Tracks cycles per turn
    exit_reason: Optional[str]
```

### Pillar 3: 4-Dimensional LLM Judge & Targeted Gap Correction
* **Strict Evaluation Rubric**: Evaluates *Completeness*, *Technical Depth*, *Structure*, and *Factual Grounding & Citations*.
* **Dynamic Gap-Driven Search**: If the Judge deducts points for missing proof or data, the graph loops back to the **Web Search Agent**, which generates queries targeting those exact gaps.
* **Target Quality Gate**: `TARGET_SCORE = 8.5` (out of 10).
* **Plateau Detection**: If $\Delta \text{score} < 0.2$, halts early to prevent token waste.

### Pillar 4: Progressive Summary-Buffer Memory (`MemorySaver` + Compactor)
* Compiled with `checkpointer = MemorySaver()`.
* Invoked with `config = {"configurable": {"thread_id": f"session_{timestamp}"}}`.
* **Memory Compactor Pre-Node**: When active buffer exceeds `BUFFER_LIMIT = 4`, older turns are compressed into a rolling `summary` (< 150 words).
* **Dual-Tier Context**: Researcher receives both long-term executive summary and short-term dialogue buffer.

### Pillar 5: Real-Time Web Grounding (`core/tools.py`)
* Dual-engine web search: Native Google Search Grounding via Gemini API (`tools: [{"google_search": {}}]`) with automatic DuckDuckGo fallback.
* Formatter attaches a verified `### 🌐 Verified Web Sources & Citations` section with markdown hyperlinks.

### Pillar 6: Continuous Session Execution Logger (`core/logger.py`)
* Persists session execution details, scorecards, and web search evidence to `logs/execution_logs_<timestamp>.txt`.

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

### ✅ Milestone A: Real-Time Web Grounding & Dynamic Search (COMPLETED)
* Dedicated `web_search` node with Google Search Grounding & DuckDuckGo fallback.
* Dynamic feedback loop: LLM Judge gaps trigger targeted searches during revision cycles.
* Formatter attaches verified markdown hyperlinks to original web sources.

### 🔜 Milestone B: Human-in-the-Loop (Approval & Steering Interrupts)
* **Goal**: Pause execution before the Formatter publishes the final report, allowing the user to approve or inject custom guidance.
* **Pattern**: `workflow.compile(checkpointer=checkpointer, interrupt_before=["formatter"])`.

### 🔜 Milestone C: Persistent Disk Checkpointer (SQLite / PostgreSQL)
* **Goal**: Persist memory across terminal restarts so sessions survive after exiting `main.py`.
* **Pattern**: Replace `MemorySaver` with `SqliteSaver.from_conn_string("research_history.db")`.

---

## 🎯 5. Job Interview Concepts Mastered in this Project

1. **State Reducers (`operator.add`)**: Prevents list overwriting; accumulates messages across turns.
2. **Cyclic Graphs**: Native cycle support in LangGraph vs. linear DAG constraints in traditional pipelines.
3. **Evaluator-Optimizer Pattern**: Structured Pydantic/JSON rubrics over subjective evaluations.
4. **Moving Goalpost Mitigation**: Using numerical thresholds and plateau detection to guarantee termination.
5. **Decoupled Agent Engineering**: Separating interface (`main.py`), execution runtime (`core/graph.py`), and tool/client layers (`core/llm.py`).
