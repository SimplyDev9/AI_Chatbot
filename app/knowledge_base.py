# app/knowledge_base.py

import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR
from app.logger import logger


class KnowledgeBase:

    def __init__(self):
        """Initialize embeddings and Chroma vector DB."""
        try:
            self.embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                region_name=os.getenv("AWS_DEFAULT_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            self.vectordb = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings
            )

            logger.info("KnowledgeBase initialized")

        except Exception:
            logger.exception("KB Init Error")
            raise

    def search(self, query: str, k: int = 10):
        try:
            logger.debug("Searching KB: %s", query)

            results = self.vectordb.similarity_search_with_score(query, k=k)

            docs = []

            for doc, score in results:
                logger.info(
                    "Score: %s | File: %s",
                    score,
                    doc.metadata.get("filename")
                )

                docs.append({
                    "text": doc.page_content,
                    "score": float(score),  # ✅ ensure numeric
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"),
                    "source_type": doc.metadata.get("source_type", "unknown"),
                    "source_url": doc.metadata.get("source_url")
                })

            if not docs:
                logger.warning("No documents retrieved for query: %s", query)
                return []

            # ✅ IMPORTANT: DO NOT SORT HERE
            # Let retriever handle ranking

            return docs

        except Exception:
            logger.exception("KB Search Error for query: %s", query)
            return []

    def file_exists_in_db(vectordb, filename):
        try:
            results = vectordb.get(where={"filename": filename})
            return len(results.get("ids", [])) > 0
        except Exception as e:
            print(f"❌ DB check error: {str(e)}")
            return False
