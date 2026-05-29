"""
app/routers/voice_router.py
────────────────────────────
Voice endpoints — Speech-to-Text (Amazon Transcribe) and Text-to-Speech (Amazon Polly).

Guardrail pipeline is injected AFTER transcription and BEFORE the RAG query,
using the same dual-layer (Comprehend + Bedrock Guardrail) used by the chat endpoint.
This ensures voice input receives identical policy enforcement to text input.

    Audio → Transcribe → [GUARDRAIL CHECK] → RAG → Polly → MP3
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.dependencies import get_current_user
from app.guardrails import run_guardrails

logger = logging.getLogger(__name__)

router = APIRouter()

# ── AWS config ───────────────────────────────────────────────────────────────
AWS_REGION           = os.environ.get("AWS_REGION", "us-east-1")
TRANSCRIBE_S3_BUCKET = os.environ.get("TRANSCRIBE_S3_BUCKET", "")

_transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
_polly_client      = boto3.client("polly",      region_name=AWS_REGION)
_s3_client         = boto3.client("s3",         region_name=AWS_REGION) if TRANSCRIBE_S3_BUCKET else None


# ════════════════════════════════════════════════════════════════════════════
# Helper — Amazon Transcribe (batch, synchronous poll)
# ════════════════════════════════════════════════════════════════════════════

async def _transcribe_from_bytes(audio_bytes: bytes, media_format: str = "webm") -> str:
    """
    Upload audio to S3 → start Transcription job → poll until done → return transcript.
    Raises HTTPException on failure.
    """
    if not TRANSCRIBE_S3_BUCKET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "TRANSCRIBE_S3_BUCKET environment variable is not set. "
                "Provide an S3 bucket name so Amazon Transcribe can read the audio file."
            ),
        )

    job_name = f"voice-{uuid.uuid4().hex}"
    s3_key   = f"transcribe-input/{job_name}.{media_format}"
    s3_uri   = f"s3://{TRANSCRIBE_S3_BUCKET}/{s3_key}"

    # 1. Upload raw audio to S3
    try:
        _s3_client.put_object(
            Bucket=TRANSCRIBE_S3_BUCKET,
            Key=s3_key,
            Body=audio_bytes,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 upload failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"S3 upload error: {exc}")

    # 2. Start Transcription job
    try:
        _transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=media_format,
            LanguageCode="en-US",
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("Transcribe start failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Transcribe start error: {exc}")

    # 3. Poll until COMPLETED or FAILED (max 25 s)
    deadline = time.time() + 25
    while time.time() < deadline:
        await asyncio.sleep(1.5)
        try:
            resp = _transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        except (BotoCoreError, ClientError) as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        job_status = resp["TranscriptionJob"]["TranscriptionJobStatus"]

        if job_status == "COMPLETED":
            transcript_uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(transcript_uri)
                r.raise_for_status()
                data = r.json()

            transcript_text = data["results"]["transcripts"][0]["transcript"]

            try:
                _s3_client.delete_object(Bucket=TRANSCRIBE_S3_BUCKET, Key=s3_key)
            except Exception:
                pass

            return transcript_text

        if job_status == "FAILED":
            reason = resp["TranscriptionJob"].get("FailureReason", "Unknown")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Transcription failed: {reason}",
            )

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail="Transcription timed out. Please try again with a shorter recording.",
    )


# ════════════════════════════════════════════════════════════════════════════
# Helper — Amazon Polly
# ════════════════════════════════════════════════════════════════════════════

def _polly_synthesize(text: str, voice_id: str = "Ruth") -> bytes:
    """Call Polly SynthesizeSpeech and return raw MP3 bytes."""
    truncated = text[:3000]
    try:
        response = _polly_client.synthesize_speech(
            Text=truncated,
            OutputFormat="mp3",
            VoiceId=voice_id,
            Engine="Generative",
        )
    except _polly_client.exceptions.TextLengthExceededException:
        response = _polly_client.synthesize_speech(
            Text=truncated[:1500],
            OutputFormat="mp3",
            VoiceId=voice_id,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("Polly synthesis failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Polly TTS error: {exc}")

    return response["AudioStream"].read()


# ════════════════════════════════════════════════════════════════════════════
# Route models
# ════════════════════════════════════════════════════════════════════════════

class SynthesizeRequest(BaseModel):
    text:     str
    voice_id: str = "Joanna"


# ════════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/transcribe",
    summary="Speech-to-Text via Amazon Transcribe + Guardrail check",
    response_model=dict,
)
async def transcribe_audio(
        audio: UploadFile = File(..., description="Audio recording (WebM/Opus from browser MediaRecorder)"),
        _user=Depends(get_current_user),
):
    """
    Accept an audio file, transcribe it via Amazon Transcribe, run the
    transcript through the dual-layer guardrail pipeline, then return it.

    The guardrail check happens HERE (not in /chat) so the frontend can
    display the blocked-message banner before attempting the RAG query.

    Returns:
        { "transcript": "what the user said" }

    Raises HTTP 400 with { "code": "CONTENT_BLOCKED", "message": "..." }
    if either guardrail layer flags the transcribed text.
    """
    audio_bytes = await audio.read()

    if len(audio_bytes) < 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is too small. Please record at least 1 second of audio.",
        )

    # Determine format from content-type or filename
    content_type = audio.content_type or ""
    if "ogg" in content_type or (audio.filename or "").endswith(".ogg"):
        media_format = "ogg"
    elif "mp4" in content_type or (audio.filename or "").endswith(".mp4"):
        media_format = "mp4"
    else:
        media_format = "webm"

    # 1. Transcribe
    transcript = await _transcribe_from_bytes(audio_bytes, media_format=media_format)

    if not transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not detect speech. Please speak clearly and try again.",
        )

    # 2. ── Dual-layer guardrail check on the transcribed text ────────────
    is_blocked, user_message = run_guardrails(transcript)

    if is_blocked:
        logger.warning(
            "Guardrail blocked voice input | user=%s | transcript=%r",
            getattr(_user, "email", "unknown"),
            transcript[:120],
        )
        # Return 400 with the same structured body the chat endpoint uses,
        # so the frontend can handle both identically.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code":       "CONTENT_BLOCKED",
                "message":    user_message,
                "transcript": transcript,  # include so FE can still display what was said
            },
        )

    return {"transcript": transcript}


@router.post(
    "/synthesize",
    summary="Text-to-Speech via Amazon Polly",
    response_class=StreamingResponse,
)
async def synthesize_speech(body: SynthesizeRequest, _user=Depends(get_current_user)):
    """
    Accept JSON body with `text` and optional `voice_id`,
    call Amazon Polly, and stream MP3 audio back to the client.
    """
    if not body.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text must not be empty.",
        )

    mp3_bytes = await asyncio.get_event_loop().run_in_executor(
        None, _polly_synthesize, body.text, body.voice_id
    )

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )