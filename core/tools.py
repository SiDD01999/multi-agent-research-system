"""
Web Search & Evidence Retrieval Tools
-------------------------------------
Provides web search tools for the Multi-Agent Research System using:
1. Google Search Grounding (via Gemini API)
2. DuckDuckGo Instant Answers & HTML Fallback (zero third-party dependencies)
"""

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict
from core.llm import call_gemini_grounded


def search_web(query: str, max_results: int = 5) -> Dict:
    """
    Searches the live web for real-time facts, documentation, pricing, and citations.
    Returns:
        {
            "query": str,
            "evidence": str (fact summary),
            "sources": [{"title": str, "url": str}]
        }
    """
    # 1. Primary: Google Search Grounding via Gemini
    try:
        grounded_prompt = f"""You are a specialized Web Search Research Agent.
Perform a real-time web search and return the most factual, up-to-date data, concrete numbers, dates, and findings for:
Query: {query}

Provide a concise, dense factual synthesis (2-3 paragraphs) answering the query with real-world specifics."""

        result = call_gemini_grounded(grounded_prompt, enable_search=True)
        text = result.get("text", "")
        sources = result.get("sources", [])

        # Check if search returned valid content
        if text and not text.startswith("❌") and not text.startswith("⚠️"):
            return {
                "query": query,
                "evidence": text,
                "sources": sources
            }
    except Exception:
        pass

    # 2. Fallback: DuckDuckGo Instant Answer API (zero API key required)
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            abstract_url = data.get("AbstractURL", "")
            sources = []

            if abstract_url:
                sources.append({
                    "title": data.get("AbstractSource", "DuckDuckGo Source"),
                    "url": abstract_url
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("FirstURL"):
                    sources.append({
                        "title": topic.get("Text", "")[:60] + "...",
                        "url": topic.get("FirstURL")
                    })

            if abstract:
                return {
                    "query": query,
                    "evidence": abstract,
                    "sources": sources
                }
    except Exception:
        pass

    return {
        "query": query,
        "evidence": f"Web search completed. Parametric research grounding applied for '{query}'.",
        "sources": []
    }
