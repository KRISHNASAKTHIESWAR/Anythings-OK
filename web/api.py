"""
web/api.py
FastAPI bridge between the GraphRAG Python backend and the Next.js frontend.
Run from the project root:
    uvicorn web.api:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
import uuid
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Bootstrap: add project root to path and load .env ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── Now safe to import project modules ─────────────────────────────────────────
from graphdb.ingest import ingest_file          # noqa: E402
from graphdb.model import graphdb               # noqa: E402
from graphdb.retriever import retrieve_and_answer  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Upload directory ───────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job tracker ──────────────────────────────────────────────────────
# job_id → {"status": "pending|processing|done|error", "doc_id": str|None, "error": str|None}
jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for blocking ingest calls
executor = ThreadPoolExecutor(max_workers=2)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="GraphRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents():
    """Return all ingested documents from Neo4j."""
    try:
        docs = graphdb.list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and its orphaned graph nodes from Neo4j."""
    try:
        graphdb.delete_document(doc_id)
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def get_stats():
    """Return graph statistics."""
    try:
        return graphdb.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Upload + async ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _run_ingest(job_id: str, file_path: str):
    """Background worker: run ingest_file and update job status."""
    jobs[job_id]["status"] = "processing"
    try:
        doc_id = ingest_file(file_path)
        jobs[job_id].update({"status": "done", "doc_id": doc_id})
        logger.info(f"[JOB {job_id}] Done. doc_id={doc_id}")
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})
        logger.error(f"[JOB {job_id}] Failed: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except Exception:
            pass


@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a file upload and kick off background ingestion.
    Returns a job_id immediately — poll /api/upload/status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"

    # Save upload to disk
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register job
    jobs[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "doc_id": None,
        "error": None,
    }

    # Schedule background ingest
    background_tasks.add_task(_run_ingest, job_id, str(dest))

    logger.info(f"[UPLOAD] job_id={job_id}, file={file.filename}")
    return {"job_id": job_id, "filename": file.filename, "status": "pending"}


@app.get("/api/upload/status/{job_id}")
def upload_status(job_id: str):
    """Poll ingestion job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# Chat — SSE streaming
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    model: str = "qwen3-small-ctx"
    hops: int = 2


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Stream chat response via Server-Sent Events (SSE).
    The retriever returns a generator of text tokens.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_stream():
        loop = asyncio.get_event_loop()
        try:
            # retrieve_and_answer with stream=True returns a generator
            # Run blocking call in thread pool to avoid blocking the event loop
            gen = await loop.run_in_executor(
                executor,
                lambda: retrieve_and_answer(query, model=request.model, stream=False, hops=request.hops),
            )
            # For non-streaming, yield the whole response
            token = gen if isinstance(gen, str) else "".join(gen)
            yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[CHAT] Error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    async def streaming_event_stream():
        loop = asyncio.get_event_loop()
        try:
            # Run blocking generator in a thread
            def _get_gen():
                return retrieve_and_answer(
                    query, model=request.model, stream=True, hops=request.hops
                )

            gen = await loop.run_in_executor(executor, _get_gen)

            # The generator itself is blocking — iterate in thread
            queue: asyncio.Queue = asyncio.Queue()

            def _consume():
                try:
                    for token in gen:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, f"__ERR__{exc}")

            loop.run_in_executor(executor, _consume)

            while True:
                token = await queue.get()
                if token is None:
                    yield "data: [DONE]\n\n"
                    break
                if isinstance(token, str) and token.startswith("__ERR__"):
                    yield f"data: [ERROR] {token[7:]}\n\n"
                    break
                # Escape newlines for SSE
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"

        except Exception as e:
            logger.error(f"[CHAT] Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        streaming_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}
