#app/logger.py
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json

# Configuration via environment (safe defaults)
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.getcwd(), "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_LOG_BYTES = int(os.getenv("MAX_LOG_BYTES", 5 * 1024 * 1024))  # 5 MB
BACKUP_COUNT = int(os.getenv("BACKUP_COUNT", 5))

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
log_filename = os.path.join(LOG_DIR, f"chatbot_logs_{datetime.now().strftime('%Y%m%d')}.txt")

# Logging configuration
logger = logging.getLogger("chatbot_logger")

# Only configure handlers once to avoid duplicate logs if module is reloaded/imported multiple times
if not logger.handlers:
    # Set logger level
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # File handler (rotating)
    file_handler = RotatingFileHandler(log_filename, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# Prevent log messages from being propagated to the root logger (avoids duplicate output)
logger.propagate = False


def log_interaction(query, response, context, agent_type):
    logger.info(f"Agent: {agent_type}\nQuery: {query}\nResponse: {response}\nContext: {context}\n")


def log_ingest(message: str = None, **meta):
    """Custom log for ingestion process.

    Usage:
    - log_ingest("Message string")  # backwards compatible
    - log_ingest(message="File ingested", filename="a.pdf", status="ingested", source="sharepoint")

    This emits an info-level log with a JSON-like metadata payload appended for easier parsing.
    """

    if message and not meta:
        # Backwards-compatible simple message
        logger.info(message)
        return

    # Build a structured payload
    payload = {"message": message or "ingest_event", **meta}

    # Log as a JSON string to make it easy to parse in files; keep level INFO
    logger.info("INGEST: %s", json.dumps(payload, ensure_ascii=False))
