# app/prompt_builder.py

def build_prompt(query: str, documents):
    try:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are a highly accurate AI assistant. You MUST answer the user's question ONLY using the provided context below.

==============================
CONTEXT:
{context}
==============================

USER QUESTION:
{query}

INSTRUCTIONS:

1. GREETING RULE (HIGHEST PRIORITY):
- If the user question is a greeting (e.g., "hi", "hello", "hey", "good morning", "how are you", etc.):
  - Respond ONLY with a polite greeting.
  - DO NOT add anything else.
  - DO NOT reference the context.
  - DO NOT explain limitations.
  - STOP after greeting.

2. OTHERWISE (RAG MODE):
- Use ONLY the information from the context.
- Do NOT use any external knowledge.
- If the answer is clearly present, provide a precise and complete answer.
- If the answer is NOT present in the context, respond exactly with:
  "I don't have enough information in the knowledge base to answer this."
- Do NOT guess or hallucinate.

3. Keep the answer clear and concise.

ANSWER:
"""

        return prompt

    except Exception as e:
        print(f"❌ Prompt Builder Error: {str(e)}")
        return query