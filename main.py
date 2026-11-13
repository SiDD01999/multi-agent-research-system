import os
from core.graph import run_research
from core.logger import SessionLogger, set_current_logger


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    print("=" * 60)
    print("🤖 Multi-Agent Research System | Powered by LangGraph & Gemini")
    print("=" * 60)

    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY is not set in .env")
    else:
        print("🔑 API Key configured.")

    # Start session logger
    session_logger = SessionLogger()
    set_current_logger(session_logger)

    # Unique thread_id for this CLI session (used by MemorySaver)
    thread_id = f"session_{session_logger.timestamp}"
    print(f"📁 [Session Logger]: Recording session to logs/{session_logger.filename}")
    print(f"🧠 [MemorySaver]: Thread ID active: '{thread_id}'\n")

    print("Type your prompt and press Enter. Type 'exit' or 'q' to quit.\n")

    try:
        while True:
            try:
                prompt = input("👉 Enter prompt: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session...")
                break

            if not prompt:
                continue

            if prompt.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\nThinking...")
            answer = run_research(prompt, thread_id=thread_id)
            print("\n--- Answer ---")
            print(answer)
            print("-" * 60 + "\n")

    finally:
        session_logger.end_session()
        print(f"📁 [Session Logger]: Session closed -> logs/{session_logger.filename}\n")


if __name__ == "__main__":
    main()
