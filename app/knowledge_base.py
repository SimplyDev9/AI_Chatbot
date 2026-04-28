import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR
from app.logger import logger
from rank_bm25 import BM25Okapi


class KnowledgeBase:

    def __init__(self):
        """Initialize embeddings and Chroma vector DB."""
        try:
            self.embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v2:0",
                region_name=os.getenv("AWS_DEFAULT_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                model_kwargs={
                    "dimensions": 512,
                    "normalize": True
                }
            )

            self.vectordb = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings
            )

            # ✅ Build BM25 index
            self._build_bm25_index()

            logger.info("KnowledgeBase initialized")

        except Exception:
            logger.exception("KB Init Error")
            raise

    # ============================================================
    # ✅ VECTOR SEARCH
    # ============================================================

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
                    "score": float(score),
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"),
                    "source_type": doc.metadata.get("source_type", "unknown"),
                    "source_url": doc.metadata.get("source_url")
                })

            if not docs:
                logger.warning("No documents retrieved for query: %s", query)
                return []

            # ❌ Do NOT sort here — retriever handles it
            return docs

        except Exception:
            logger.exception("KB Search Error for query: %s", query)
            return []

    # ============================================================
    # ✅ BM25 INDEX BUILD
    # ============================================================

    def _build_bm25_index(self):
        try:
            data = self.vectordb.get()

            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            self.bm25_docs = []
            self.bm25_metadata = []

            # ✅ Ensure alignment + safety
            for doc, meta in zip(documents, metadatas):
                if doc and isinstance(doc, str):
                    self.bm25_docs.append(doc)
                    self.bm25_metadata.append(meta if meta else {})

            if not self.bm25_docs:
                logger.warning("BM25 corpus is empty")
                self.bm25 = None
                return

            tokenized_corpus = [
                doc.lower().split()
                for doc in self.bm25_docs
            ]

            self.bm25 = BM25Okapi(tokenized_corpus)

            logger.info("BM25 index built successfully with %d docs", len(self.bm25_docs))

        except Exception:
            logger.exception("BM25 index build failed")
            self.bm25 = None

    # ============================================================
    # ✅ BM25 SEARCH
    # ============================================================

    def bm25_search(self, query: str, k: int = 5):
        try:
            if not self.bm25:
                logger.warning("BM25 not initialized")
                return []

            tokenized_query = query.lower().split()

            scores = self.bm25.get_scores(tokenized_query)

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]

            results = []

            for i in top_indices:
                results.append({
                    "text": self.bm25_docs[i],
                    "metadata": self.bm25_metadata[i],
                    "score": float(scores[i])  # BM25 score
                })

            return results

        except Exception:
            logger.exception("BM25 search error")
            return []

    # ============================================================
    # ✅ UTILITY
    # ============================================================

    def file_exists_in_db(self, filename):
        try:
            results = self.vectordb.get(where={"filename": filename})
            return len(results.get("ids", [])) > 0
        except Exception:
            logger.exception("DB check error")
            return False