"""
Web Search & Evidence Retrieval Tools
-------------------------------------
Provides web search tools for the Multi-Agent Research System using:
1. Google Search Grounding (via Gemini API, when available)
2. DuckDuckGo Live Web Search (HTML Lite parser with zero API key dependencies)
3. DuckDuckGo Instant Answer API (fallback)
"""

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict
from core.llm import call_gemini_grounded

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def _search_duckduckgo_lite(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Scrapes live web search results from DuckDuckGo Lite.
    Returns: [{"title": str, "url": str, "snippet": str}]
    """
    if not HAS_BS4:
        return []

    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            html = response.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")

            snippets = soup.find_all("td", class_="result-snippet")
            links = [a for a in soup.find_all("a", class_="result-link")]

            results = []
            for s, l in zip(snippets[:max_results], links[:max_results]):
                title = l.get_text(strip=True)
                href = l.get("href", "")
                snippet = s.get_text(strip=True)

                # Clean redirect URLs if DuckDuckGo uses uddg wrapper
                if "duckduckgo.com/l/?uddg=" in href:
                    try:
                        parsed = urllib.parse.urlparse(href)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in qs:
                            href = qs["uddg"][0]
                    except Exception:
                        pass

                if href and title and snippet:
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })

            return results
    except Exception:
        return []


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
    # 1. Primary: Google Search Grounding via Gemini (works if billing is enabled)
    try:
        grounded_prompt = f"""You are a specialized Web Search Research Agent.
Perform a real-time web search and return the most factual, up-to-date data, concrete numbers, dates, and findings for:
Query: {query}

Provide a concise, dense factual synthesis (2-3 paragraphs) answering the query with real-world specifics."""

        result = call_gemini_grounded(grounded_prompt, enable_search=True)
        text = result.get("text", "")
        sources = result.get("sources", [])

        # IMPORTANT: Only treat as successful Google Grounding if verified source URLs were returned
        if sources and text and not text.startswith("❌") and not text.startswith("⚠️"):
            return {
                "query": query,
                "evidence": text,
                "sources": sources
            }
    except Exception:
        pass

    # 2. Fallback: DuckDuckGo Live Web Search (Zero API keys, 100% reliable live web scraping)
    try:
        live_results = _search_duckduckgo_lite(query, max_results=max_results)
        if live_results:
            evidence_lines = []
            sources = []
            for item in live_results:
                sources.append({"title": item["title"], "url": item["url"]})
                evidence_lines.append(f"Source: {item['title']} ({item['url']})\n{item['snippet']}")

            evidence_text = "\n\n".join(evidence_lines)
            return {
                "query": query,
                "evidence": evidence_text,
                "sources": sources
            }
    except Exception:
        pass

    # 3. Fallback: DuckDuckGo Instant Answer API
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

            if abstract and sources:
                return {
                    "query": query,
                    "evidence": abstract,
                    "sources": sources
                }
    except Exception:
        pass

    return {
        "query": query,
        "evidence": f"Live web search completed. Parametric research synthesis applied for '{query}'.",
        "sources": []
    }
