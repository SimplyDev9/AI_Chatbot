import os

from app.knowledge_base import KnowledgeBase
from app.logger import logger

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# ✅ Load once (same model, lighter stack)
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
model.eval()


def rerank_pairs(pairs):
    queries = [q for q, _ in pairs]
    docs = [d for _, d in pairs]

    inputs = tokenizer(
        queries,
        docs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    return scores.tolist()


def retrieve_documents(query: str, k: int = None):
    if k is None:
        k = int(os.getenv("RETRIEVER_TOP_K", "3"))
    try:
        kb = KnowledgeBase()
        logger.debug("Retrieving documents for: %s", query)

        # ============================================================
        # Step 1: Hybrid Retrieval
        # ============================================================
        vector_results = kb.search(query, k=15)
        bm25_results = kb.bm25_search(query, k=10)

        combined_results = vector_results + bm25_results

        if not combined_results:
            logger.warning("No documents found for query: %s", query)
            return []

        # ============================================================
        # Step 2: Deduplication
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
        # Step 3: Cross-Encoder Reranking
        # ============================================================
        pairs = [(query, d["text"]) for d in unique_docs]
        rerank_scores = rerank_pairs(pairs)

        for i in range(len(unique_docs)):
            unique_docs[i]["rerank_score"] = float(rerank_scores[i])

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
        # Step 4: Absolute Score Floor Filter (read from env)
        # ============================================================
        if not unique_docs:
            return []

        min_rerank_threshold = float(os.getenv("MIN_RERANK_SCORE", "-3.0"))

        unique_docs = [
            doc for doc in unique_docs
            if doc["rerank_score"] >= min_rerank_threshold
        ]

        if not unique_docs:
            logger.warning(
                "All documents rejected by absolute rerank threshold (%.2f) for query: %s",
                min_rerank_threshold, query
            )
            return []

        # ============================================================
        # Step 5: Relative Score Drop Filter + Top-K (read from env)
        # ============================================================
        best_score = unique_docs[0]["rerank_score"]
        max_score_drop = float(os.getenv("MAX_RERANK_SCORE_DROP", "4.0"))
        score_threshold = best_score - max_score_drop

        final_docs = [
            doc for doc in unique_docs
            if doc["rerank_score"] >= score_threshold
        ]

        logger.info(
            "Relative Filter → Best: %.2f | Threshold: %.2f | Kept: %d / %d docs",
            best_score, score_threshold, len(final_docs), len(unique_docs)
        )

        # ============================================================
        # Step 6: Context Cleaning — keep top sentences per chunk
        # ============================================================

        def clean_chunk(text: str, query: str, max_sentences: int = 2) -> str:
            sentences = re.split(r'(?<=[.!?])\s+', text)

            if len(sentences) <= max_sentences:
                return text

            pairs = [(query, s) for s in sentences]

            try:
                scores = rerank_pairs(pairs)
            except Exception:
                return " ".join(sentences[:max_sentences])

            ranked = sorted(
                zip(sentences, scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_sentences = [s for s, _ in ranked[:max_sentences]]
            return " ".join(top_sentences)

        cleaned_docs = []

        for doc in final_docs[:k]:
            cleaned_text = clean_chunk(doc["text"], query)
            cleaned_docs.append({**doc, "text": cleaned_text})

        return cleaned_docs

    except Exception:
        logger.exception("Retriever error for query: %s", query)
        return []