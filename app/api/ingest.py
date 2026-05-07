from fastapi import APIRouter, Query, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from pathlib import Path
import os
import shutil
import time

from app.ingest_corpus import ingest, ingest_single_file
from app.config import CHROMA_DIR
from app.logger import log_ingest
from app.rag import get_vectordb
from app.sharepoint_ingestion import ingest_from_sharepoint
from app.core.dependencies import require_permission

router = APIRouter()

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "corpus"))


class IngestRequest(BaseModel):
    clear: bool = False


@router.post("/ingest")
def ingest_corpus(
        req: IngestRequest,
        user=Depends(require_permission("ingest")),
):
    ingest(clear=req.clear)
    return {"status": "success", "message": "Corpus ingested successfully."}


@router.delete("/clear_db")
def clear_db(user=Depends(require_permission("ingest"))):
    if not os.path.exists(CHROMA_DIR):
        return {"status": "skipped", "message": "Chroma DB not found."}

    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(CHROMA_DIR)
            return {"status": "success", "message": "Chroma DB cleared."}
        except PermissionError:
            time.sleep(1)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Could not clear DB after retries."}


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


@router.get("/list_docs")
def list_docs(user=Depends(require_permission("ingest"))):
    vectordb = get_vectordb()
    data = vectordb.get(include=["metadatas"])

    filenames = list({
        meta.get("filename")
        for meta in data.get("metadatas", [])
        if meta and "filename" in meta
    })

    return {"count": len(filenames), "files": filenames}


@router.post("/upload_doc")
async def upload_doc(
        file: UploadFile = File(...),
        user=Depends(require_permission("ingest")),
):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = CORPUS_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        ingest_single_file(file_path)
        return {
            "status": "success",
            "message": f"{file.filename} uploaded and ingested successfully",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ingestion failed: {str(e)}",
        }


@router.post("/ingest_sharepoint")
def ingest_sharepoint(
        site_id: str,
        folder_path: str,
        user=Depends(require_permission("ingest")),
):
    try:
        ingest_from_sharepoint(site_id)
        return {"status": "success", "message": "SharePoint files ingested successfully"}
    except Exception as e:
        return {"status": "error", "message": f"SharePoint ingestion failed: {str(e)}"}