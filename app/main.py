# app/main.py
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chatbot import answer_query
from app.ingest_corpus import ingest
from app.config import CHROMA_DIR
from app.rag import get_vectordb
from app.logger import log_ingest, logger
from pathlib import Path
from app.ingest_corpus import ingest_single_file
import uvicorn
import os
import shutil
import time
from dotenv import load_dotenv
from app.sharepoint_ingestion import ingest_from_sharepoint
from app.sharepoint_loader import get_access_token
import requests
from typing import Any, cast

load_dotenv()

logger.info("main.py loaded successfully")

app = FastAPI(title="AI Chatbot API", version="1.0")
# CORPUS_DIR = Path("Python_Project/corpus")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "corpus"))

# Prepare middleware class in a way that satisfies type checkers
middleware_cls: Any = CORSMiddleware

# ============================================
# CORS
# ============================================
app.add_middleware(
    middleware_cls,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # type: ignore

# ============================================
# Request Models
# ============================================

class ChatRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    clear: bool = False

class SiteRequest(BaseModel):
    hostname: str
    site_name: str


@app.post("/get_site_id")
def get_site_id(data: SiteRequest):
    token = get_access_token()


    url = f"https://graph.microsoft.com/v1.0/sites/{data.hostname}:/sites/{data.site_name}"

    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)

    return response.json()


# ============================================
# Health
# ============================================

@app.get("/health")
def health_check():
    logger.info("/health endpoint hit")
    return {"status": "ok", "message": "AI Chatbot API is running 🚀"}


@app.get("/")
def home():
    return {"message": "AI Chatbot API is running 🚀"}


# ============================================
# Chat
# ============================================

@app.post("/chat")
def chat(req: ChatRequest):
    result = answer_query(req.query)

    return {
        "query": req.query,
        "response": result["response"],
        "sources": result["sources"],
        "retrieved_context": result["retrieved_context"]
    }


# ============================================
# Ingest
# ============================================

@app.post("/ingest")
def ingest_corpus(req: IngestRequest):
    ingest(clear=req.clear)
    return {"status": "success", "message": "Corpus ingested successfully."}


# ============================================
# Clear Entire DB
# ============================================

@app.delete("/clear_db")
def clear_db():
    if not os.path.exists(CHROMA_DIR):
        return {"status": "skipped", "message": "Chroma DB not found."}

    log_ingest(message="attempt_clear_db", action="clear_db", status="started", dir=str(CHROMA_DIR))

    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(CHROMA_DIR)
            log_ingest(message="clear_db_success", action="clear_db", status="success", dir=str(CHROMA_DIR))
            return {"status": "success", "message": "Chroma DB cleared."}
        except PermissionError as e:
            log_ingest(message="clear_db_locked", action="clear_db", status="locked", attempt=attempt + 1, error=str(e))
            time.sleep(1)
        except Exception as e:
            log_ingest(message="clear_db_failed", action="clear_db", status="failed", error=str(e))
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": f"Could not clear {CHROMA_DIR} after {max_retries} retries."}


# ============================================
# Modern Delete (Metadata Based)
# ============================================

@app.delete("/delete_doc")
def delete_doc(filename: str = Query(..., description="Exact filename to delete from Chroma DB")):
    """
    Deletes embeddings for a specific document using metadata filter.
    No rebuild. No folder deletion. No Windows lock issues.
    """
    try:
        log_ingest(message="delete_request", action="delete_doc", filename=filename)

        vectordb = get_vectordb()

        # Check existence
        results = vectordb.get(where={"filename": filename})

        if not results or not results.get("ids"):
            log_ingest(message="not_found", action="delete_doc", filename=filename, status="not_found")
            raise HTTPException(
                status_code=404,
                detail=f"{filename} not found in vector DB."
            )

        # Delete by metadata
        vectordb.delete(where={"filename": filename})

        # Release DB handle (important on Windows)
        vectordb = None

        log_ingest(message="delete_success", action="delete_doc", filename=filename, status="deleted")

        return {
            "status": "success",
            "message": f"Deleted embeddings for {filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        log_ingest(f"❌ Error deleting {filename}: {e}")
        return {"status": "error", "message": str(e)}


# ============================================
# List Documents
# ============================================

@app.get("/list_docs")
def list_docs():
    try:
        vectordb = get_vectordb()
        log_ingest(message="list_docs", action="list_docs", status="started")

        data = vectordb.get(include=["metadatas"])

        filenames = list({
            meta.get("filename")
            for meta in data.get("metadatas", [])
            if meta and "filename" in meta
        })

        vectordb = None

        log_ingest(message="list_docs_success", action="list_docs", count=len(filenames))

        return {"count": len(filenames), "files": filenames}

    except Exception as e:
        log_ingest(f"❌ Failed to fetch document list: {e}")
        return {"status": "error", "message": str(e)}


# ============================================
# Upload & Ingest
# ============================================

@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)

        file_path = CORPUS_DIR / file.filename

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        log_ingest(message="file_uploaded", action="upload_doc", filename=str(file_path), status="saved")

        ingest_single_file(file_path)
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded and ingested successfully."
        }

    except Exception as e:
        log_ingest(message="upload_failed", action="upload_doc", filename=file.filename, error=str(e))
        return {"status": "error", "message": str(e)}


@app.post("/ingest_sharepoint")
def ingest_sharepoint(site_id: str, folder_path: str):
    ingest_from_sharepoint(site_id, folder_path)

    return {
        "status": "success",
        "message": "SharePoint files ingested successfully"
    }


def start():
    """
    Start the FastAPI server programmatically
    """
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )


if __name__ == "__main__":
    start()
