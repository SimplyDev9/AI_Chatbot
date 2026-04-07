from app.knowledge_base import KnowledgeBase
from app.logger import logger
from sentence_transformers import CrossEncoder

# ✅ Load once (global)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_documents(query: str, k: int = 3):
    try:
        kb = KnowledgeBase()
        logger.debug("Retrieving documents for: %s", query)

        # ✅ Step 1: Get more candidates
        results = kb.search(query, k=10)

        if not results:
            logger.warning("No documents found for query: %s", query)
            return []

        # ✅ Step 2: Prepare for reranking
        pairs = []
        docs = []

        for doc in results:
            text = doc.get("text", "")
            pairs.append((query, text))

            docs.append({
                "text": text,
                "metadata": doc.get("metadata"),
                "score": float(doc.get("score", 9999))
            })

        # ✅ Step 3: Cross-encoder scoring
        rerank_scores = reranker.predict(pairs)

        for i in range(len(docs)):
            docs[i]["rerank_score"] = float(rerank_scores[i])

        # ✅ Step 4: Sort by rerank score (HIGHER = better)
        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

        logger.info("After reranking:")

        for d in docs[:5]:
            logger.info(
                "Rerank Score: %s | File: %s",
                d["rerank_score"],
                d["metadata"].get("filename")
            )

        # ============================================================
        # ✅ Step 5: Confidence Filtering (NO LLM CALL)
        # ============================================================

        if not docs:
            return []

        best_score = docs[0]["rerank_score"]

        second_score = docs[1]["rerank_score"] if len(docs) > 1 else None

        # ✅ Score gap (how confident we are)
        score_gap = best_score - second_score if second_score else best_score

        logger.info(
            "Confidence Check → Best: %s | Second: %s | Gap: %s",
            best_score, second_score, score_gap
        )

        # ============================================================
        # ✅ Confidence filtering (clean & correct)
        # ============================================================

        if second_score is None:
            logger.info("Only one document found → accepting result")
        else:
            score_gap = best_score - second_score

        logger.info(
            "Confidence Check → Best: %s | Second: %s | Gap: %s",
            best_score, second_score, score_gap
        )

        if score_gap < 0.5:
            logger.warning("Rejected: Low confidence gap")
            return []

        # ============================================================
        # ✅ Step 6: Return top-k (clean & confident)
        # ============================================================

        docs = docs[:k]

        return docs

    except Exception:
        logger.exception("Retriever error for query: %s", query)
        return []