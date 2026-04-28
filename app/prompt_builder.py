# app/prompt_builder.py

from app.logger import logger


def build_prompt(query: str, documents):
    try:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are a precise AI assistant. Answer the user's question using ONLY the information provided in the context below. Never reference the context, document, or source in your answer — respond as if you simply know the answer.

==============================
CONTEXT:
{context}
==============================

USER QUESTION:
{query}

RULES:
1. If the question is a greeting, respond with a short polite greeting only and nothing else.
2. Answer directly and concisely using only what is stated in the context above.
3. Do not begin your answer with phrases like "According to", "Based on", "The context says", or any similar phrasing.
4. Do not infer, assume, or use any knowledge beyond what is in the context.
5. If the answer is not present in the context, respond with exactly: "I don't have enough information in the knowledge base to answer this."

ANSWER:"""

        return prompt

    except Exception:
        logger.exception("Prompt Builder Error")
        return query