# 🤖 Autonomous Multi-Agent Research System

AI-powered system that analyzes any website using coordinated agents.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add API Key
Get free key from: https://console.groq.com

Create `.env`:
```
GROQ_API_KEY=your_key_here
```

### 3. Configure
Edit `config.py`:
```python
TARGET_WEBSITE = "www.your-site.com"
WEBSITE_CONTENT = "Paste content here..."
USER_QUERY = "What to analyze..."
```

### 4. Run
```bash
python run_simple.py
```

## Agents

| Agent | Focus |
|-------|-------|
| 🔍 Business Analyst | Market gaps, monetization |
| 🏗️ IT Architect | AI features, tech roadmap |
| 📝 Proposal Writer | Executive proposal |

## Files

| File | Purpose |
|------|---------|
| `config.py` | ⭐ All inputs here |
| `run_simple.py` | Main script |
| `.env` | API key |
| `output/` | Generated proposals |

## Tech Stack
- Groq (Llama 3.3 70B) - Free tier
- Python 3.9+
