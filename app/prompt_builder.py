# app/prompt_builder.py

def build_prompt(query: str, documents):
    try:
        context = "\n\n".join([doc["text"] for doc in documents])

        prompt = f"""
You are a highly accurate AI assistant.

You MUST answer ONLY using the provided context.

CONTEXT:
{context}

QUESTION:
{query}

RULES:
- Use ONLY the context
- Do NOT guess
- If answer not present, say:
"I don't have enough information in the knowledge base to answer this."

ANSWER:
"""

        return prompt

    except Exception as e:
        print(f"❌ Prompt Builder Error: {str(e)}")
        return query