import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import CHROMA_DIR
from app.core.dependencies import require_permission
from app.ingest_corpus import ingest, ingest_single_file
from app.knowledge_base import invalidate_knowledge_base
from app.logger import log_ingest
from app.rag import get_vectordb
from app.sharepoint_ingestion import ingest_from_sharepoint

router = APIRouter()

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "corpus"))

# ─────────────────────────────────────────────────────────────
# In-memory job store  (job_id → asyncio.Queue of SSE events)
# ─────────────────────────────────────────────────────────────
_jobs: dict[str, asyncio.Queue] = {}


def _make_job() -> tuple[str, asyncio.Queue]:
    job_id = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue()
    _jobs[job_id] = q
    return job_id, q


def _cleanup_job(job_id: str):
    _jobs.pop(job_id, None)


# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────
# /ingest  (bulk corpus re-ingest — no SSE needed, rare admin op)
# ─────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    clear: bool = False


@router.post("/ingest")
def ingest_corpus(
        req: IngestRequest,
        user=Depends(require_permission("ingest")),
):
    ingest(clear=req.clear)
    return {"status": "success", "message": "Corpus ingested successfully."}


# ─────────────────────────────────────────────────────────────
# DELETE /clear_db
# ─────────────────────────────────────────────────────────────
@router.delete("/clear_db")
def clear_db(user=Depends(require_permission("ingest"))):
    if not os.path.exists(CHROMA_DIR):
        return {"status": "skipped", "message": "Chroma DB not found."}

    for attempt in range(5):
        try:
            shutil.rmtree(CHROMA_DIR)
            return {"status": "success", "message": "Chroma DB cleared."}
        except PermissionError:
            time.sleep(1)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Could not clear DB after retries."}


# ─────────────────────────────────────────────────────────────
# DELETE /delete_doc
# ─────────────────────────────────────────────────────────────
@router.delete("/delete_doc")
def delete_doc(
        filename: str = Query(...),
        user=Depends(require_permission("ingest")),
):
    vectordb = get_vectordb()
    results = vectordb.get(where={"filename": filename})

    if not results or not results.get("ids"):
        raise HTTPException(status_code=404, detail="File not found")

    vectordb.delete(where={"filename": filename})
    return {"status": "success", "message": f"Deleted {filename}"}


# ─────────────────────────────────────────────────────────────
# GET /list_docs
# ─────────────────────────────────────────────────────────────
@router.get("/list_docs")
def list_docs(user=Depends(require_permission("ingest"))):
    from datetime import datetime

    vectordb = get_vectordb()
    data = vectordb.get(include=["metadatas"])

    unique_filenames = list({
        meta.get("filename")
        for meta in data.get("metadatas", [])
        if meta and "filename" in meta
    })

    files = []
    for filename in unique_filenames:
        file_path = CORPUS_DIR / filename
        size = None
        uploaded_on = None
        if file_path.exists():
            stat = file_path.stat()
            size = stat.st_size
            uploaded_on = datetime.fromtimestamp(stat.st_mtime).isoformat()
        files.append({"filename": filename, "size": size, "uploaded_on": uploaded_on})

    return {"count": len(files), "files": files}


# ─────────────────────────────────────────────────────────────
# POST /upload_doc  →  returns {job_id, filename}
# GET  /upload_doc/progress/{job_id}  →  SSE stream
# ─────────────────────────────────────────────────────────────
# Allowed file types — matches the frontend acceptedTypes list
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".pptx", ".ppt", ".csv", ".xlsx"}
_MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


@router.post("/upload_doc")
async def upload_doc(
        file: UploadFile = File(...),
        user=Depends(require_permission("ingest")),
):
    # ── 1. Extension whitelist ────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {suffix!r} is not allowed. "
                   f"Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    # ── 2. Sanitise filename — strip any path traversal components ────────────
    # Path traversal: an attacker could upload a file named "../../etc/passwd"
    # Path(name).name strips all directory components, keeping only the basename.
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # ── 3. Read into memory and enforce size limit ────────────────────────────
    # We read before writing so we can reject oversized files without
    # leaving a partial file on disk.
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({size_mb:.1f} MB) exceeds the {_MAX_FILE_SIZE_MB} MB limit."
        )

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4. Write sanitised file to disk ──────────────────────────────────────
    file_path = CORPUS_DIR / safe_name
    file_path.write_bytes(content)

    # Create job + start background ingestion
    job_id, queue = _make_job()
    loop = asyncio.get_event_loop()

    def _progress(stage: str, percent: int, message: str):
        """Called from worker thread — thread-safe queue put."""
        asyncio.run_coroutine_threadsafe(
            queue.put({"stage": stage, "percent": percent, "message": message}),
            loop,
        )

    async def _run():
        try:
            await loop.run_in_executor(
                None,
                lambda: ingest_single_file(file_path, progress=_progress),
            )
            # Ensure a final done event even if callback already sent one
            if not queue.empty():
                return
            invalidate_knowledge_base()  # force BM25 rebuild on next query
            await queue.put({"stage": "done", "percent": 100,
                             "message": f"{file.filename} ingested successfully"})
        except Exception as e:
            await queue.put({"stage": "error", "percent": 0, "message": str(e)})

    asyncio.create_task(_run())

    return {"job_id": job_id, "filename": file.filename}


@router.get("/upload_doc/progress/{job_id}")
async def upload_progress(
        job_id: str,
        user=Depends(require_permission("ingest")),
):
    queue = _jobs.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Job not found or already completed")

    async def _stream():
        # Heartbeat every 15 s so the connection doesn't time out on large files
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield _sse({"stage": "heartbeat", "percent": -1,
                                "message": "Processing..."})
                    continue

                yield _sse(event)

                if event.get("stage") in ("done", "error"):
                    _cleanup_job(job_id)
                    break
        except asyncio.CancelledError:
            _cleanup_job(job_id)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )


# ─────────────────────────────────────────────────────────────
# POST /ingest_sharepoint  →  returns {job_id}
# GET  /ingest_sharepoint/progress/{job_id}  →  SSE stream
# ─────────────────────────────────────────────────────────────
@router.post("/ingest_sharepoint")
async def ingest_sharepoint_endpoint(
        site_id: str,
        folder_path: str,
        user=Depends(require_permission("ingest")),
):
    job_id, queue = _make_job()
    loop = asyncio.get_event_loop()

    def _progress(stage: str, percent: int, message: str):
        asyncio.run_coroutine_threadsafe(
            queue.put({"stage": stage, "percent": percent, "message": message}),
            loop,
        )

    async def _run():
        try:
            await loop.run_in_executor(
                None,
                lambda: ingest_from_sharepoint(site_id, progress=_progress),
            )
            if not queue.empty():
                return
            invalidate_knowledge_base()  # force BM25 rebuild on next query
            await queue.put({"stage": "done", "percent": 100,
                             "message": "SharePoint ingestion completed"})
        except Exception as e:
            await queue.put({"stage": "error", "percent": 0, "message": str(e)})

    asyncio.create_task(_run())
    return {"job_id": job_id}


@router.get("/ingest_sharepoint/progress/{job_id}")
async def sharepoint_progress(
        job_id: str,
        user=Depends(require_permission("ingest")),
):
    queue = _jobs.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Job not found or already completed")

    async def _stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield _sse({"stage": "heartbeat", "percent": -1,
                                "message": "Processing SharePoint files..."})
                    continue

                yield _sse(event)

                if event.get("stage") in ("done", "error"):
                    _cleanup_job(job_id)
                    break
        except asyncio.CancelledError:
            _cleanup_job(job_id)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )