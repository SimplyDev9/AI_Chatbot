#app.ingest_corpus.py
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import docx2txt
from PyPDF2 import PdfReader
from app.config import CHROMA_DIR
import hashlib
from app.document_loader import load_document
from app.logger import log_ingest
from pptx import Presentation
import pandas as pd
import pdfplumber
import re
from app.logger import logger

load_dotenv()


def safe_clear_chroma_db():
    """Safely clears the Chroma DB directory (handles Windows file locks)."""
    if not os.path.exists(CHROMA_DIR):
        log_ingest(message="no_chroma_db", action="safe_clear", status="skipped", dir=str(CHROMA_DIR))
        return

    log_ingest(message="attempt_clear_old_chroma", action="safe_clear", status="started", dir=str(CHROMA_DIR))
    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(CHROMA_DIR)
            log_ingest(message="cleared_old_chroma", action="safe_clear", status="success", dir=str(CHROMA_DIR))
            return
        except PermissionError as e:
            log_ingest(message="chroma_locked", action="safe_clear", status="locked", attempt=attempt + 1, error=str(e))
            time.sleep(1)
        except Exception as e:
            log_ingest(message="clear_chroma_failed", action="safe_clear", status="failed", error=str(e))
            break
    log_ingest(message="clear_chroma_retry_failed", action="safe_clear", status="failed", attempts=max_retries)


def extract_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    text = ""

    try:
        if ext == ".txt":
            text = fp.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".docx":
            text = docx2txt.process(str(fp))

        elif ext == ".pdf":
            pages_text = []
            with pdfplumber.open(str(fp)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            page_text += "\n" + " | ".join(cleaned_row)
                    pages_text.append(page_text)
            text = ("\n".join(pages_text))

        elif ext == ".csv":
            df = pd.read_csv(fp, encoding="utf-8", encoding_errors="ignore")
            text = df.to_string()

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(fp)
            text = df.to_string()

        elif ext == ".pptx":
            presentation = Presentation(str(fp))
            slides_text = []
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slides_text.append(shape.text)
            text = "\n".join(slides_text)

        else:
            log_ingest(message="unsupported_file_type", action="extract_text", filename=fp.name, error=f"No extractor for {ext}")

    except Exception as e:
        log_ingest(message="failed_read", action="extract_text", filename=fp.name, error=str(e))

    return text.strip()

def ingest(corpus_dir=None, clear=False):
    if corpus_dir is None:
        corpus_dir = Path("Python_Project/corpus").resolve()
    else:
        corpus_dir = Path(corpus_dir).resolve()

    if not corpus_dir.exists():
        log_ingest(message="corpus_not_found", action="ingest", dir=str(corpus_dir))
        return

    # ✅ Use the safe clear function
    if clear:
        safe_clear_chroma_db()

    # Initialize embeddings and Chroma DB
    embed_model = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.getenv("AWS_DEFAULT_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        model_kwargs={
            "dimensions": 512,
            "normalize": True
        }
    )

    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embed_model
    )

    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embed_model
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    file_paths = list(corpus_dir.glob("*"))

    for fpath in file_paths:

        file_hash = calculate_file_hash(fpath)

        # Check if file already exists
        existing = vectordb.get(where={"filename": fpath.name})

        if existing and existing.get("metadatas"):
            existing_hash = existing["metadatas"][0].get("file_hash")

            if existing_hash == file_hash:
                log_ingest(message="skip_unchanged", action="ingest", filename=fpath.name, status="skipped")
                continue

            else:
                log_ingest(message="file_changed", action="ingest", filename=fpath.name, status="updating")
                vectordb.delete(where={"filename": fpath.name})

        text = load_document(str(fpath))

        if not text:
            continue

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks, start=1):
            vectordb.add_texts(
                texts=[chunk],
                metadatas=[{
                    "filename": fpath.name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "local",
                    "file_type": fpath.suffix.lower(),
                    "file_size": os.path.getsize(fpath),
                }]
            )

        log_ingest(message="file_ingested", action="ingest", filename=fpath.name, chunks=len(chunks))


def calculate_file_hash(file_path: Path):
    """Generate hash for file to detect changes"""
    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()


def ingest_single_file(file_path: Path, metadata=None):
    try:
        logger.info("[INGEST] Starting ingestion for file: %s", file_path.name)

        logger.info("[INGEST] Initializing embedding model")
        embed_model = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            model_kwargs={
                "dimensions": 512,
                "normalize": True
            }
        )

        logger.info("[INGEST] Connecting to vector store at: %s", str(CHROMA_DIR))
        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed_model
        )

        logger.info("[INGEST] Computing file hash for: %s", file_path.name)
        file_hash = calculate_file_hash(file_path)
        logger.info("[INGEST] File hash: %s", file_hash)

        existing = vectordb.get(where={"filename": file_path.name})
        if existing and existing.get("metadatas"):
            existing_hash = existing["metadatas"][0].get("file_hash")
            if existing_hash == file_hash:
                logger.info("[INGEST] File unchanged, skipping re-ingestion: %s", file_path.name)
                log_ingest(message="file_unchanged_skipped", action="ingest_single_file", filename=file_path.name)
                return
            logger.info("[INGEST] File changed, deleting old chunks for: %s", file_path.name)
            vectordb.delete(where={"filename": file_path.name})

        logger.info("[INGEST] Extracting text from: %s (type: %s)", file_path.name, file_path.suffix.lower())
        text = extract_text(file_path)
        logger.info("[INGEST] Extracted %d characters from: %s", len(text), file_path.name)

        chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        logger.info("[INGEST] Splitting text — chunk_size=%d, chunk_overlap=%d", chunk_size, chunk_overlap)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        logger.info("[INGEST] Created %d chunks for: %s", len(chunks), file_path.name)

        default_metadata = {
            "filename": file_path.name,
            "source_type": "upload",
            "source_url": None
        }
        final_metadata = {**default_metadata, **(metadata or {})}

        metadatas = [
            {
                "filename": final_metadata["filename"],
                "file_hash": file_hash,
                "chunk_index": i,
                "source_type": final_metadata["source_type"],
                "source_url": final_metadata.get("source_url") or "",
                "file_type": file_path.suffix.lower(),
            }
            for i, _ in enumerate(chunks)
        ]

        logger.info("[INGEST] Embedding and storing %d chunks into vector DB", len(chunks))
        vectordb.add_texts(texts=chunks, metadatas=metadatas)
        logger.info("[INGEST] Successfully ingested %d chunks for: %s", len(chunks), file_path.name)

        log_ingest(message="file_ingested_metadata", action="ingest_single_file", filename=file_path.name, metadata=final_metadata)

    except Exception as e:
        logger.error("[INGEST] Failed to ingest file: %s | Error: %s", file_path.name, str(e))
        log_ingest(message="error_ingesting_file", action="ingest_single_file", filename=file_path.name, error=str(e))