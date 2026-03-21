import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory (AI_Chatbot)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Paths for Chroma DB and corpus
CORPUS_DIR = os.path.join(BASE_DIR, "Python_Project", "corpus")

CHROMA_DIR = "Python_Project/chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Example: SentenceTransformer model

# ------------------------
# Logging Config
# ------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)