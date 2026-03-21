#app/logger.py
import os
import logging
from datetime import datetime

# Ensure logs directory exists
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_filename = os.path.join(log_dir, f"chatbot_logs_{datetime.now().strftime('%Y%m%d')}.txt")

# Logging configuration
logger = logging.getLogger("chatbot_logger")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def log_interaction(query, response, context, agent_type):
    logger.info(f"Agent: {agent_type}\nQuery: {query}\nResponse: {response}\nContext: {context}\n")

def log_ingest(message):
    """Custom log for ingestion process"""
    logger.info(message)
