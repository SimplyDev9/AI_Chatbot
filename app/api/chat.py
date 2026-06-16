"""
app/api/chat.py  — OPTIMIZED
──────────────────────────────
Changes from previous version:

  1. DB logging moved to BackgroundTasks (no longer in the request path)
     Before: response time = guardrails + LLM + DB write
     After:  response time = guardrails + LLM   (DB write is async, user never waits)

  2. Each background task creates its OWN DB session (safe — avoids session
     closed / detached instance errors when the request session closes first)

  3. chatbot.py timing pattern shown at the bottom — add this to your
     answer_query() function to populate retrieval/llm/embedding columns
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.chatbot import answer_query
from app.core.dependencies import get_current_user
from app.db.database import SessionLocal          # ← used by background tasks
from app.guardrails import run_guardrails
from app.db.dashboard_models import ChatLog, GuardrailLog

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query:      str       = Field(..., min_length=1, max_length=4_000)
    department: str | None = Field(None, max_length=100)


class ChatResponse(BaseModel):
    response: str
    sources:  list = []


# ─────────────────────────────────────────────────────────────────────────────
# Background task helpers — each creates its own session so they are safe
# to run AFTER the request session has already closed.
# ─────────────────────────────────────────────────────────────────────────────

def _bg_log_chat(
    *,
    user_id, user_email, query, response_preview,
    status, blocked_by=None, source_document=None,
    response_time_ms=None, retrieval_time_ms=None,
    llm_generation_time_ms=None, embedding_time_ms=None,
    grounding_score=None, department=None,
):
    """Runs in background — opens and closes its own DB session."""
    db = SessionLocal()
    try:
        log = ChatLog(
            user_id=user_id, user_email=user_email,
            query=query, response_preview=response_preview,
            status=status, blocked_by=blocked_by,
            source_document=source_document,
            response_time_ms=response_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            llm_generation_time_ms=llm_generation_time_ms,
            embedding_time_ms=embedding_time_ms,
            grounding_score=grounding_score,
            department=department,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error("Background chat_log write failed: %s", e)
        db.rollback()
    finally:
        db.close()


def _bg_log_guardrail(*, user_email, user_id, query, layer, block_type, action="BLOCKED"):
    """Runs in background — opens and closes its own DB session."""
    logger.warning(
        "WRITING GUARDRAIL LOG | layer=%s | type=%s",
        layer,
        block_type,
    )
    db = SessionLocal()
    try:
        log = GuardrailLog(
            user_id=user_id, user_email=user_email,
            query_preview=query[:120],
            layer=layer, block_type=block_type, action=action,
        )
        db.add(log)
        db.commit()
        logger.warning(
            "GUARDRAIL LOG COMMITTED | layer=%s | type=%s",
            layer,
            block_type,
        )
    except Exception as e:
        logger.error("Background guardrail_log write failed: %s", e)
        db.rollback()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    body:             ChatRequest,
    background_tasks: BackgroundTasks,               # ← inject background tasks
    current_user=Depends(get_current_user),
    # NOTE: No db: Session dependency here — background tasks create their own
):
    raw_query  = body.query.strip()
    user_email = getattr(current_user, "email", "unknown")
    user_id    = getattr(current_user, "id",    None)
    t_start    = time.monotonic()

    if not raw_query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # ── Guardrail check ───────────────────────────────────────────────────
    is_blocked, user_message = run_guardrails(raw_query)

    if is_blocked:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        blocked_by = _detect_layer(user_message)
        block_type = _detect_block_type(user_message)

        logger.warning(
            "Guardrail blocked | user=%s | layer=%s | type=%s | preview=%r",
            user_email, blocked_by, block_type, raw_query[:120],
        )

        # ── Fire-and-forget: logs happen AFTER response is returned ──────
        _bg_log_guardrail(
            user_email=user_email,
            user_id=user_id,
            query=raw_query,
            layer=blocked_by,
            block_type=block_type,
            action="BLOCKED",
        )

        _bg_log_chat(
            user_id=user_id,
            user_email=user_email,
            query=raw_query,
            response_preview=user_message,
            status="blocked",
            blocked_by=blocked_by,
            response_time_ms=elapsed_ms,
            department=body.department,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CONTENT_BLOCKED", "message": user_message},
        )

    # ── RAG chain ─────────────────────────────────────────────────────────
    try:
        result     = answer_query(raw_query)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        response_text = result.get("response", "")
        sources       = result.get("sources",  [])
        first_source  = sources[0].get("name", "") if sources else None

        # ── Fire-and-forget: user gets response immediately ───────────────
        background_tasks.add_task(
            _bg_log_chat,
            user_id=user_id, user_email=user_email,
            query=raw_query,
            response_preview=response_text[:500],
            status="success",
            source_document=first_source,
            response_time_ms=elapsed_ms,
            # These are None until chatbot.py returns them (see PATCH 2 below)
            retrieval_time_ms=result.get("retrieval_time_ms"),
            llm_generation_time_ms=result.get("llm_time_ms"),
            embedding_time_ms=result.get("embedding_time_ms"),
            grounding_score=result.get("grounding_score"),
            department=body.department,
        )

        return ChatResponse(response=response_text, sources=sources)

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.exception("answer_query failed: %s", exc)

        background_tasks.add_task(
            _bg_log_chat,
            user_id=user_id, user_email=user_email,
            query=raw_query, response_preview=str(exc)[:200],
            status="failed", response_time_ms=elapsed_ms,
            department=body.department,
        )
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_layer(message: str) -> str:
    msg = message.lower()
    if "inappropriate language" in msg or "threatening" in msg: return "L0"
    if "profanity" in msg or "hate speech" in msg or "sexual" in msg: return "L1"
    if "social security" in msg or "credit" in msg or "password" in msg or "sensitive" in msg: return "L2"
    return "L3a"


def _detect_block_type(message: str) -> str:
    msg = message.lower()
    if "profanity" in msg:               return "PROFANITY"
    if "hate speech" in msg:             return "HATE_SPEECH"
    if "threatening" in msg:             return "THREAT"
    if "insulting" in msg:               return "INSULT"
    if "social security" in msg:         return "SSN"
    if "credit" in msg or "debit" in msg:return "CREDIT_DEBIT_NUMBER"
    if "password" in msg:                return "PASSWORD"
    if "aws" in msg and "key" in msg:    return "AWS_ACCESS_KEY"
    if "sensitive" in msg:               return "PII"
    if "content policy" in msg:         return "DENIED_TOPIC"
    return "JAILBREAK"


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 2 — chatbot.py timing instrumentation
#
# Add time.monotonic() measurements around each step of answer_query().
# Share your chatbot.py and we will apply these exactly.
# The pattern is:
#
#   import time
#
#   def answer_query(query: str) -> dict:
#
#       # ── Step 1: Embed the query ───────────────────────────────
#       t_embed = time.monotonic()
#       embedded_query = embeddings.embed_query(query)     # your embed call
#       embedding_ms = int((time.monotonic() - t_embed) * 1000)
#
#       # ── Step 2: Retrieve documents ────────────────────────────
#       t_ret = time.monotonic()
#       docs = retriever.get_relevant_documents(query)     # your retrieve call
#       retrieval_ms = int((time.monotonic() - t_ret) * 1000)
#
#       # ── Step 3: LLM generation ────────────────────────────────
#       t_llm = time.monotonic()
#       answer = llm.invoke(prompt)                        # your LLM call
#       llm_ms = int((time.monotonic() - t_llm) * 1000)
#
#       return {
#           "response":           answer,
#           "sources":            sources,
#           # ── NEW keys (chat.py reads these) ──
#           "retrieval_time_ms":  retrieval_ms,
#           "llm_time_ms":        llm_ms,
#           "embedding_time_ms":  embedding_ms,
#           "grounding_score":    grounding_score,   # float 0-1 if available
#       }
#
# Share chatbot.py and we'll apply this precisely.
# ═══════════════════════════════════════════════════════════════════════════