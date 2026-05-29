"""
app/api/chat.py
───────────────
POST /chat  — main RAG chat endpoint.

Guardrail pipeline is injected BEFORE the query reaches the LLM:
  1. AWS Comprehend  (toxicity / hate-speech / profanity)
  2. AWS Bedrock Guardrail  (custom organisational policy / prompt-injection)

Both layers fail-open so an AWS outage never blocks a legitimate request.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.chatbot import answer_query
from app.core.dependencies import get_current_user
from app.guardrails import run_guardrails

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4_000)


class ChatResponse(BaseModel):
    response: str
    sources:  list = []


# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="RAG chat with dual-layer guardrails",
)
async def chat(
        body: ChatRequest,
        current_user=Depends(get_current_user),
):
    """
    Accept a user query, run it through the dual-layer guardrail pipeline,
    then forward to the RAG chain if the content is clean.

    HTTP 400 is returned with a structured body when either guardrail layer
    blocks the message:

        {
          "detail": {
            "code":    "CONTENT_BLOCKED",
            "message": "<user-facing explanation>"
          }
        }
    """
    raw_query = body.query.strip()

    if not raw_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    # ── Dual-layer guardrail check ────────────────────────────────────────
    is_blocked, user_message = run_guardrails(raw_query)

    if is_blocked:
        logger.warning(
            "Guardrail blocked request | user=%s | preview=%r",
            getattr(current_user, "email", "unknown"),
            raw_query[:120],
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code":    "CONTENT_BLOCKED",
                "message": user_message,
            },
        )

    # ── Clean — forward to RAG chain ─────────────────────────────────────
    try:
        result = answer_query(raw_query)
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources", []),
        )
    except Exception as exc:
        logger.exception("answer_query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request.",
        )