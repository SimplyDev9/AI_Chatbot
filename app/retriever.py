from app.knowledge_base import KnowledgeBase
from app.logger import logger
from sentence_transformers import CrossEncoder

# ✅ Load once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_documents(query: str, k: int = 3):

    try:
        kb = KnowledgeBase()
        logger.debug("Retrieving documents for: %s", query)

        # ============================================================
        # ✅ Step 1: Hybrid Retrieval
        # ============================================================
        vector_results = kb.search(query, k=10)
        bm25_results = kb.bm25_search(query, k=5)

        combined_results = vector_results + bm25_results

        if not combined_results:
            logger.warning("No documents found for query: %s", query)
            return []

        # ============================================================
        # ✅ Step 2: Deduplication (CRITICAL)
        # ============================================================
        seen = set()
        unique_docs = []

        for doc in combined_results:
            text = doc.get("text")

            if text and text not in seen:
                seen.add(text)
                unique_docs.append({
                    "text": text,
                    "metadata": doc.get("metadata", {}),
                    "score": float(doc.get("score", 9999))
                })


        # ============================================================
        # ✅ Step 3: Cross-Encoder Reranking
        # ============================================================
        pairs = [(query, d["text"]) for d in unique_docs]
        rerank_scores = reranker.predict(pairs)

        for i in range(len(unique_docs)):
            unique_docs[i]["rerank_score"] = float(rerank_scores[i])

        # Sort by rerank score (HIGHER = better)
        unique_docs = sorted(
            unique_docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        logger.info("After reranking:")
        for d in unique_docs[:5]:
            logger.info(
                "Rerank Score: %s | File: %s",
                d["rerank_score"],
                d["metadata"].get("filename")
            )

        # ============================================================
        # ✅ Step 4: Confidence Filtering (NO LLM)
        # ============================================================
        if len(unique_docs) < 1:
            return []

        best_score = unique_docs[0]["rerank_score"]
        second_score = (
            unique_docs[1]["rerank_score"]
            if len(unique_docs) > 1 else None
        )

        if second_score is not None:
            score_gap = best_score - second_score

            logger.info(
                "Confidence Check → Best: %s | Second: %s | Gap: %s",
                best_score, second_score, score_gap
            )

            if score_gap < 0.5:
                logger.warning("Rejected: Low confidence gap")
                return []

        # ============================================================
        # ✅ Step 5: Return Top-K
        # ============================================================
        return unique_docs[:k]

    except Exception:
        logger.exception("Retriever error for query: %s", query)
        return []