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

load_dotenv()


def safe_clear_chroma_db():
    """Safely clears the Chroma DB directory (handles Windows file locks)."""
    if not os.path.exists(CHROMA_DIR):
        log_ingest("⚠️ No existing Chroma DB found, skipping clear.")
        return

    log_ingest(f"🧹 Attempting to clear old Chroma DB at {CHROMA_DIR}...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(CHROMA_DIR)
            log_ingest(f"✅ Successfully cleared old Chroma DB at {CHROMA_DIR}")
            return
        except PermissionError as e:
            log_ingest(f"⚠️ Chroma DB locked (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)
        except Exception as e:
            log_ingest(f"❌ Failed to clear Chroma DB: {e}")
            break
    log_ingest(f"🚫 Could not delete Chroma DB after {max_retries} retries.")


def extract_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    text = ""

    try:
        if ext == ".txt":
            text = fp.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".docx":
            text = docx2txt.process(str(fp))

        elif ext == ".pdf":
            reader = PdfReader(str(fp))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".csv":
            df = pd.read_csv(fp)
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

    except Exception as e:
        log_ingest(f"⚠️ Failed to read {fp.name}: {e}")

    return text.strip()


def ingest(corpus_dir=None, clear=False):
    if corpus_dir is None:
        corpus_dir = Path("Python_Project/corpus").resolve()
    else:
        corpus_dir = Path(corpus_dir).resolve()

    if not corpus_dir.exists():
        log_ingest(f"❌ Corpus folder not found: {corpus_dir}")
        return

    # ✅ Use the safe clear function
    if clear:
        safe_clear_chroma_db()

    # Initialize embeddings and Chroma DB
    embed_model = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=os.getenv("AWS_DEFAULT_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
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
                log_ingest(f"⏭️ Skipping unchanged file: {fpath.name}")
                continue

            else:
                log_ingest(f"♻️ File changed. Updating: {fpath.name}")
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

        log_ingest(f"✅ {fpath.name} ingested with {len(chunks)} chunks")


def calculate_file_hash(file_path: Path):
    """Generate hash for file to detect changes"""
    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()


def ingest_single_file(file_path: Path, metadata=None):
    try:
        embed_model = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed_model
        )

        file_hash = calculate_file_hash(file_path)

        existing = vectordb.get(where={"filename": file_path.name})

        if existing and existing.get("metadatas"):
            vectordb.delete(where={"filename": file_path.name})

        text = extract_text(file_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        chunks = splitter.split_text(text)

        default_metadata = {
            "filename": file_path.name,
            "source_type": "upload",
            "source_url": None
        }

        final_metadata = {**default_metadata, **(metadata or {})}

        for i, chunk in enumerate(chunks):
            vectordb.add_texts(
                texts=[chunk],
                metadatas=[{
                    "filename": final_metadata["filename"],
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "source_type": final_metadata["source_type"],   # ✅ CRITICAL
                    "source_url": final_metadata["source_url"],     # ✅ CRITICAL
                    "file_type": file_path.suffix.lower(),
                }]
            )

        print(f"✅ Ingested {file_path.name} with metadata:", final_metadata)
        print("🔥 METADATA SAVED:", final_metadata)

    except Exception as e:
        print(f"❌ Error ingesting file: {str(e)}")
