from pathlib import Path
from app.ingest_corpus import ingest_single_file
from app.sharepoint_loader import list_files_in_folder, download_file
from app.sharepoint_tracker import is_file_updated, update_tracker
from app.knowledge_base import KnowledgeBase
from app.logger import logger, log_ingest

# ── Lazy singleton ────────────────────────────────────────────────────────────
# Do NOT instantiate KnowledgeBase at module level.
# Importing sharepoint_ingestion at startup (even when SharePoint is unused)
# would attempt to connect to ChromaDB and AWS Bedrock immediately, crashing
# the entire API if either service is unavailable.
_kb: "KnowledgeBase | None" = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def file_exists_in_db(filename):
    try:
        results = _get_kb().vectordb.get(where={"filename": filename})
        return len(results.get("ids", [])) > 0
    except Exception:
        logger.exception("DB check error for: %s", filename)
        return False


def ingest_from_sharepoint(site_id, progress=None):
    """
    progress: optional callable(stage: str, percent: int, message: str)
    """
    def _emit(stage, pct, msg):
        if callable(progress):
            try:
                progress(stage, pct, msg)
            except Exception:
                pass
    _emit('fetching', 5, 'Fetching file list from SharePoint...')
    files = list_files_in_folder(site_id)
    total = len([f for f in files if 'file' in f])
    processed = 0

    temp_dir = Path("temp_sharepoint")
    temp_dir.mkdir(exist_ok=True)

    _emit('fetching', 15, f'Found {total} files to process')
    for file in files:
        try:
            if "file" not in file:
                continue

            filename = file["name"]
            file_id = file["id"]
            last_modified = file["lastModifiedDateTime"]
            download_url = file.get("@microsoft.graph.downloadUrl")
            if not download_url:
                logger.warning("No download URL for file: %s", filename)
                continue
            web_url = file.get("webUrl")

            logger.debug("File ID: %s | Last Modified: %s", file_id, last_modified)

            # 🔥 NEW LOGIC (CRITICAL FIX)
            exists_in_db = file_exists_in_db(filename)
            updated = is_file_updated(file_id, last_modified)

            if not updated and exists_in_db:
                log_ingest(message="skip_unchanged_sharepoint", action="ingest", filename=filename, file_id=file_id)
                continue

            log_ingest(message="reingest_sharepoint", action="ingest", filename=filename, file_id=file_id)

            local_path = temp_dir / filename

            # ✅ DOWNLOAD
            log_ingest(message="download_start", action="download", filename=filename, file_id=file_id)
            download_file(download_url, local_path)

            # ✅ METADATA
            metadata = {
                "filename": filename,
                "source_type": "sharepoint",
                "source_url": web_url
            }

            # ✅ OPTIONAL (BEST PRACTICE)
            # delete old chunks before re-ingest
            try:
                _get_kb().vectordb.delete(where={"filename": filename})
                log_ingest(message="deleted_old_chunks", action="cleanup", filename=filename)
            except Exception:
                logger.exception("Delete warning for filename: %s", filename)

            # ✅ INGEST
            processed += 1
            pct = 15 + int(80 * processed / max(total, 1))
            _emit('ingesting', pct, f'Ingesting {filename} ({processed}/{total})...')
            log_ingest(message="ingest_start", action="ingest", filename=filename, source="sharepoint")
            ingest_single_file(local_path, metadata=metadata)

            # ✅ UPDATE TRACKER
            update_tracker(file_id, last_modified)

        except Exception:
            logger.exception("Failed file processing for: %s", file.get('name'))
            continue

    log_ingest(message="sharepoint_ingestion_completed", action="ingest", source="sharepoint")
    _emit("done", 100, f"SharePoint ingestion complete — {processed} file(s) processed")