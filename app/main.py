# app/main.py
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Depends
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
from app.db.database import test_connection
from app.db.database import engine
from app.db.base import Base
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.database import SessionLocal
from app.db.seed import seed_roles_permissions
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.admin import router as admin_router
from app.api.auth import router as auth_router


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
# Auth
# ============================================

app.include_router(auth_router)

# ============================================
# Health
# ============================================

@app.get("/health")
def health_check():
    logger.info("/health endpoint hit")
    return {"status": "ok", "message": "AI Chatbot API is running 🚀"}


# ✅ Create tables
Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return {"message": "AI Chatbot API is running 🚀"}


@app.get("/db-test")
def db_test():
    return {"status": test_connection()}

# =========================================
# DB Seeding
# =========================================
app.include_router(admin_router)

# ============================================
# Chat
# ============================================

app.include_router(chat_router)

# ============================================
# Ingest
# ============================================


app.include_router(ingest_router)



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
