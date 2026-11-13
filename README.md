# Multi-Agent Research System

A modular multi-agent research system built with **LangGraph** and powered by **Google Gemini**.

---

## Architecture: Two Collaborative Agents

```
START ───► [ Agent 1: Researcher ] ───► [ Agent 2: Reviewer ] ───► END
```

1. **Agent 1 (`researcher_node`)**: Focuses on raw information retrieval, analysis, and deep substance. Generates `draft`.
2. **Agent 2 (`reviewer_node`)**: Reads `draft`, cleans up fluff, organizes key findings, and produces the structured `final_report`.

---

## Project Structure

```
multi-agent-research-system/
├── .env                  # API keys (GEMINI_API_KEY)
├── requirements.txt      # Project dependencies
├── main.py               # Pure CLI runner (never needs editing)
└── core/                 # Core agent logic
    ├── __init__.py
    ├── llm.py            # Gemini client & standard API caller
    └── graph.py          # LangGraph state, 2 agent nodes, edges, & compilation
```

---

## How to Run

In your terminal:
```bash
python main.py
```
