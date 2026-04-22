"""Thread-id keyed side-channel for BFDTS visualization.

The planner tool writes its structured trace here; the FastAPI SSE handler
drains it immediately after emitting the `make_science_plan` tool_result event.
Keeping the trace out of the tool's string output means the LLM doesn't carry
a large JSON blob in its context on follow-up turns.

Keyed by LangGraph thread_id (= chat session) so concurrent users don't
step on each other's traces — a previous single-slot implementation would
mix traces when two requests planned in parallel.
"""

import threading
from typing import Dict, Optional

_DEFAULT_KEY = "__default__"

_lock = threading.Lock()
_traces: Dict[str, dict] = {}


def set_trace(thread_id: Optional[str], trace: dict) -> None:
    key = thread_id or _DEFAULT_KEY
    with _lock:
        _traces[key] = trace


def pop_trace(thread_id: Optional[str]) -> Optional[dict]:
    key = thread_id or _DEFAULT_KEY
    with _lock:
        return _traces.pop(key, None)
