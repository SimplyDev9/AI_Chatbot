# app/main.py

from fastapi import FastAPI, HTTPException, Depends
from app.core.dependencies import require_permission
import os
from pathlib import Path
from typing import Any

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from app.core.limiter import limiter
from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.db.base import Base
from app.db.database import engine
from app.db.database import test_connection
from app.logger import logger
from app.sharepoint_loader import get_access_token
from app.routers.voice_router import router as voice_router

load_dotenv()

logger.info("main.py loaded successfully")

app = FastAPI(title="AI Chatbot API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "corpus"))

# ============================================
# CORS — restrict to known frontend origins
#
# allow_origins=["*"] + allow_credentials=True is prohibited by the CORS spec
# and is a CSRF attack vector. Any malicious site could make authenticated
# requests using the victim's JWT. Restrict to explicit origins only.
#
# Set ALLOWED_ORIGINS in .env:
#   ALLOWED_ORIGINS=https://app.yourcompany.com,https://admin.yourcompany.com
# For local dev, leave it blank — defaults to localhost:3000 only.
# ============================================
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000"]
    logger.warning(
        "CORS: ALLOWED_ORIGINS not set — restricting to localhost:3000 only. "
        "Set ALLOWED_ORIGINS in .env before deploying to production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)


# ============================================
# Request Models
# ============================================
class SiteRequest(BaseModel):
    hostname: str
    site_name: str


@app.post("/get_site_id")
def get_site_id(
        data: SiteRequest,
        user=Depends(require_permission("manage_users"))
):
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


# ── Database tables are managed by Alembic migrations, NOT create_all() ────
# create_all() is an antipattern for production:
#   - silently skips existing tables (cannot apply schema changes)
#   - no rollback capability, no migration history
#   - race condition when two instances start simultaneously
#
# Run migrations before starting the server:
#   alembic upgrade head
#
# In Docker: CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app ..."]


@app.get("/")
def home():
    return {"message": "AI Chatbot API is running 🚀"}


@app.get("/db-test")
def db_test(user=Depends(require_permission("manage_users"))):
    return {"status": test_connection()}


# =========================================
# Admin + Chat + Ingest
# =========================================
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(ingest_router)


# =========================================
# VOICE
# =========================================
app.include_router(voice_router, prefix="/voice", tags=["voice"])


def start():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=int(os.getenv("WORKERS", "1")),
    )


if __name__ == "__main__":
    start()