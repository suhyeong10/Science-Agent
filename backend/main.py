import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.patches import apply_patches

apply_patches()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.upload import router as upload_router

app = FastAPI(title="Sci-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("SCI_AGENT_BACKEND_PORT", "8787"))
    reload = os.environ.get("SCI_AGENT_RELOAD", "0") == "1"
    workers = int(os.environ.get("SCI_AGENT_WORKERS", "1"))
    # Bind to localhost only — the Next.js server on the same host proxies /api/*
    # to us, so we don't need external access.
    # Multiple workers = multiple Python processes sharing memory.db via SQLite.
    # Each worker loads its own agent; BFDTS trace dict is per-process but keyed
    # by thread_id so requests sticking to one worker are fine. uvicorn doesn't
    # support reload + workers > 1 together.
    run_kwargs = {
        "host": "127.0.0.1",
        "port": port,
    }
    if workers > 1 and not reload:
        run_kwargs["workers"] = workers
    else:
        run_kwargs["reload"] = reload
    uvicorn.run("backend.main:app", **run_kwargs)
