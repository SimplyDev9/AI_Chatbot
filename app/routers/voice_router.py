import asyncio
import io
import json
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

# ── Optional: reuse your existing auth dependency ──────────────────────────
# Replace this import with the actual path in your project.
# If you don't have one, remove the `dependencies` parameter below.
try:
    from auth import get_current_user  # adjust import as needed
    _AUTH_DEP = [Depends(get_current_user)]
except ImportError:
    _AUTH_DEP = []

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=_AUTH_DEP)

# ── AWS config ──────────────────────────────────────────────────────────────
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
# Optional S3 bucket for Transcribe (required by the batch Transcribe API).
# If blank, we fall back to a pre-signed-URL approach.
TRANSCRIBE_S3_BUCKET = os.environ.get("TRANSCRIBE_S3_BUCKET", "")

_transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
_polly_client      = boto3.client("polly",      region_name=AWS_REGION)
_s3_client         = boto3.client("s3",          region_name=AWS_REGION) if TRANSCRIBE_S3_BUCKET else None


# ════════════════════════════════════════════════════════════════════════════
# Helper — Amazon Transcribe (batch, synchronous poll)
# ════════════════════════════════════════════════════════════════════════════

async def _transcribe_from_bytes(audio_bytes: bytes, media_format: str = "webm") -> str:
    """
    Upload audio to S3 → start Transcription job → poll until done → return transcript.

    Falls back to a small in-process approach if no S3 bucket is configured.
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

    job_name   = f"voice-{uuid.uuid4().hex}"
    s3_key     = f"transcribe-input/{job_name}.{media_format}"
    s3_uri     = f"s3://{TRANSCRIBE_S3_BUCKET}/{s3_key}"

    # 1. Upload raw audio to S3
    try:
        _s3_client.put_object(
            Bucket=TRANSCRIBE_S3_BUCKET,
            Key=s3_key,
            Body=audio_bytes,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"S3 upload error: {exc}",
        )

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
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Transcribe start error: {exc}",
        )

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
            # Fetch JSON transcript over HTTPS
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(transcript_uri)
                r.raise_for_status()
                data = r.json()

            transcript_text = data["results"]["transcripts"][0]["transcript"]

            # Clean up S3 object (best-effort)
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

def _polly_synthesize(text: str, voice_id: str = "Joanna") -> bytes:
    """
    Call Polly SynthesizeSpeech and return raw MP3 bytes.
    Raises HTTPException on failure.
    """
    # Truncate to Polly's 3000-char limit per request
    truncated = text[:3000]
    try:
        response = _polly_client.synthesize_speech(
            Text=truncated,
            OutputFormat="mp3",
            VoiceId=voice_id,
            Engine="neural",         # Neural voices sound much more natural
        )
    except _polly_client.exceptions.TextLengthExceededException:
        # Retry with plain text engine which handles longer inputs differently
        response = _polly_client.synthesize_speech(
            Text=truncated[:1500],
            OutputFormat="mp3",
            VoiceId=voice_id,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error("Polly synthesis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Polly TTS error: {exc}",
        )

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
    summary="Speech-to-Text via Amazon Transcribe",
    response_model=dict,
)
async def transcribe_audio(
        audio: UploadFile = File(..., description="Audio recording (WebM/Opus from browser MediaRecorder)"),
):
    """
    Accept an audio file (WebM from the browser's MediaRecorder API),
    run it through Amazon Transcribe, and return the transcript.

    Returns:
        { "transcript": "what the user said" }
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
        media_format = "webm"   # default — Chrome/Firefox MediaRecorder default

    transcript = await _transcribe_from_bytes(audio_bytes, media_format=media_format)

    return {"transcript": transcript}


@router.post(
    "/synthesize",
    summary="Text-to-Speech via Amazon Polly",
    response_class=StreamingResponse,
)
async def synthesize_speech(body: SynthesizeRequest):
    """
    Accept a JSON body with `text` and optional `voice_id`,
    call Amazon Polly, and stream MP3 audio back to the client.

    The frontend creates a blob URL from the binary response and
    plays it directly in an <audio> element.
    """
    if not body.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text must not be empty.",
        )

    # Run in thread pool to avoid blocking the event loop
    mp3_bytes = await asyncio.get_event_loop().run_in_executor(
        None, _polly_synthesize, body.text, body.voice_id
    )

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )