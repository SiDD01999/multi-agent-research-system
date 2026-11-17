import os
import json
import urllib.request
import urllib.error
from pathlib import Path


def load_env():
    """Load environment variables from .env located at project root"""
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


load_env()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def call_gemini(prompt: str, model: str = "gemini-flash-lite-latest", enable_search: bool = False) -> str:
    """
    Direct Gemini API call using Python standard library.
    Optionally enables native Google Search Grounding when enable_search=True.
    """
    result = call_gemini_grounded(prompt, model=model, enable_search=enable_search)
    return result.get("text", "")


def call_gemini_grounded(prompt: str, model: str = "gemini-flash-lite-latest", enable_search: bool = True) -> dict:
    """
    Calls Gemini API with Google Search Grounding enabled.
    Returns structured dictionary with generated text, search queries, and web source citations.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        return {
            "text": "❌ Error: GEMINI_API_KEY is not set. Please check your .env file.",
            "search_queries": [],
            "sources": []
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    # Attach Google Search grounding tool if enabled
    if enable_search:
        payload["tools"] = [{"google_search": {}}]

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            data = json.loads(response.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if not candidates:
                return {
                    "text": "⚠️ No response text returned by the model.",
                    "search_queries": [],
                    "sources": []
                }

            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()

            # Extract Google Search Grounding Metadata
            grounding = first_candidate.get("groundingMetadata", {})
            search_queries = grounding.get("webSearchQueries", [])
            sources = []
            seen_urls = set()

            for chunk in grounding.get("groundingChunks", []):
                web = chunk.get("web", {})
                uri = web.get("uri")
                if uri and uri not in seen_urls:
                    seen_urls.add(uri)
                    sources.append({
                        "title": web.get("title", "Web Reference"),
                        "url": uri
                    })

            return {
                "text": text,
                "search_queries": search_queries,
                "sources": sources
            }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(error_body)
            msg = err_json.get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body
        
        # If model failed due to search tool incompatibility, retry once without search
        if enable_search and "tool" in error_body.lower():
            return call_gemini_grounded(prompt, model=model, enable_search=False)

        return {
            "text": f"❌ Gemini API Error ({e.code}): {msg}",
            "search_queries": [],
            "sources": []
        }
    except Exception as e:
        return {
            "text": f"❌ Request error: {str(e)}",
            "search_queries": [],
            "sources": []
        }
