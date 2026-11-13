import os
from core.graph import run_research


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    print("=" * 60)
    print("🤖 Multi-Agent Research System | Powered by LangGraph & Gemini")
    print("=" * 60)

    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY is not set in .env")
    else:
        print("🔑 API Key configured.")

    print("\nType your prompt and press Enter. Type 'exit' or 'q' to quit.\n")

    while True:
        try:
            prompt = input("👉 Enter prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...")
        answer = run_research(prompt)
        print("\n--- Answer ---")
        print(answer)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
