import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class SessionLogger:
    """
    Session-level Logger that writes all prompt executions within a single
    terminal session to one timestamped file:
    logs/execution_logs_<YYYYMMDD_HHMMSS>.txt

    The file persists and accumulates logs until the session ends (Ctrl+C or exit).
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"execution_logs_{self.timestamp}.txt"
        self.filepath = LOGS_DIR / self.filename
        self.turn_count = 0
        self.start_time = datetime.now()

        self._write_session_header()

    def write(self, text: str):
        """Append text to the session log file"""
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _write_session_header(self):
        start_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.write("=" * 80)
        self.write("MULTI-AGENT RESEARCH SYSTEM | SESSION LOG")
        self.write(f"Session Started : {start_str}")
        self.write(f"Log File        : {self.filename}")
        self.write("=" * 80)

    def start_turn(self, prompt: str):
        """Called whenever a new user prompt is executed within the session"""
        self.turn_count += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        self.write("\n" + "#" * 80)
        self.write(f"TURN #{self.turn_count} | Started at {now_str}")
        self.write("#" * 80)
        self.write(f"\n[USER PROMPT]:\n{prompt}\n")
        self.write("-" * 80)

    def log_step(self, title: str, content: str = ""):
        """Log an intermediate agent node action or output"""
        now_str = datetime.now().strftime("%H:%M:%S")
        self.write(f"\n>>> [{now_str}] {title}")
        if content:
            self.write(content)
        self.write("-" * 80)

    def log_scorecard(self, comp: int, depth: int, struct: int, overall: float, target: float, gaps: List[str]):
        """Log the LLM Judge rubric evaluation"""
        self.write("\n[LLM JUDGE SCORECARD]:")
        self.write(f"  • Completeness:    {comp}/10")
        self.write(f"  • Technical Depth: {depth}/10")
        self.write(f"  • Structure:       {struct}/10")
        self.write(f"  • Overall Score:   {overall}/10 (Target: {target}/10)")
        if gaps:
            self.write("  • Key Gaps Identified:")
            for g in gaps:
                self.write(f"    - {g}")
        self.write("-" * 80)

    def end_turn(self, reason: str, final_score: float, revisions: int):
        """Called when a single prompt completes its multi-agent cycle"""
        now_str = datetime.now().strftime("%H:%M:%S")
        self.write(f"\n[TURN #{self.turn_count} COMPLETE at {now_str}]:")
        self.write(f"  Exit Reason     : {reason}")
        self.write(f"  Verified Score  : {final_score}/10")
        self.write(f"  Total Revisions : {revisions}")
        self.write("-" * 80)

    def end_session(self):
        """Called on session termination (Ctrl+C or exit)"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = datetime.now() - self.start_time
        total_seconds = int(duration.total_seconds())
        mins, secs = divmod(total_seconds, 60)

        self.write("\n" + "=" * 80)
        self.write("SESSION TERMINATED")
        self.write(f"Ended At       : {now_str}")
        self.write(f"Duration       : {mins}m {secs}s")
        self.write(f"Total Turns    : {self.turn_count}")
        self.write("=" * 80 + "\n")


# Alias for backward compatibility
ExecutionLogger = SessionLogger

_current_session_logger: Optional[SessionLogger] = None


def get_current_logger() -> Optional[SessionLogger]:
    global _current_session_logger
    return _current_session_logger


def set_current_logger(logger: Optional[SessionLogger]):
    global _current_session_logger
    _current_session_logger = logger
