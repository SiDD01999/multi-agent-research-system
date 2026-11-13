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


def call_gemini(prompt: str, model: str = "gemini-flash-lite-latest") -> str:
    """
    Direct Gemini API call using Python standard library.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        return "❌ Error: GEMINI_API_KEY is not set. Please check your .env file."

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

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                return text.strip()
            return "⚠️ No response text returned by the model."

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(error_body)
            msg = err_json.get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body
        return f"❌ Gemini API Error ({e.code}): {msg}"
    except Exception as e:
        return f"❌ Request error: {str(e)}"
