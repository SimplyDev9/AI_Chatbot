# app/chatbot.py

from app.llm import call_bedrock_llm
from app.logger import log_interaction
from app.retriever import retrieve_documents
from app.prompt_builder import build_prompt


def answer_query(query: str):
    try:
        print(f"🔥 Incoming query: {query}")

        docs = retrieve_documents(query)

        # ✅ Handle no docs early
        if not docs:
            response = "I don't have enough information in the knowledge base to answer this."
            log_interaction(query, response, "NO_CONTEXT", None)

            return {
                "response": response,
                "sources": []
            }

        # ✅ Build prompt
        prompt = build_prompt(query, docs)

        # ✅ Call LLM (ONLY ONCE)
        response = call_bedrock_llm(prompt)

        # ✅ Detect greeting
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        is_greeting = any(greet in query.lower() for greet in greetings)

        # ✅ Decide sources
        if is_greeting or "I don't have enough information" in response:
            sources = []
        else:
            sources = []

            for doc in docs:
                metadata = doc.get("metadata", {})

                sources.append({
                    "name": metadata.get("filename", "Unknown"),
                    "type": metadata.get("source_type", "unknown"),
                    "url": metadata.get("source_url")
                })

            # ✅ Remove duplicates (by name)
            sources = list({s["name"]: s for s in sources}.values())
            sources = sources[:1]

        log_interaction(query, response, "RAG", None)

        return {
            "response": response,
            "sources": sources,
            "highlight_text": response[:200],
            "retrieved_context":docs
        }

    except Exception as e:
        error_msg = f"❌ Error in chatbot: {str(e)}"
        print(error_msg)
        log_interaction(query, error_msg, "ERROR", None)

        return {
            "response": "Something went wrong while processing your request.",
            "sources": []
        }