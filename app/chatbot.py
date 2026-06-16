from app.llm import call_bedrock_llm
from app.logger import log_interaction, logger
from app.retriever import retrieve_documents
from app.prompt_builder import build_prompt
from app.guardrails import check_output_guardrail


def answer_query(query: str):
    try:
        logger.debug("Incoming query: %s", query)

        # ✅ Step 1: Greeting detection BEFORE retrieval
        greetings = [
            "hi", "hello", "hey", "good morning", "good evening",
            "good afternoon", "how are you", "how's it going", "greetings", "what is your name"
        ]

        query_clean = query.lower().strip()
        is_greeting = query_clean in greetings

        if is_greeting:
            response = "Hello"
            log_interaction(query, response, "GREETING", None)

            return {
                "response": response,
                "sources": [],
                "retrieved_context": []
            }

        # ✅ Step 2: Retrieve documents (NO normalization needed)
        docs = retrieve_documents(query)

        # ✅ Handle no docs
        if not docs:
            response = "I don't have enough information in the knowledge base to answer this."
            log_interaction(query, response, "NO_CONTEXT", None)

            return {
                "response": response,
                "sources": [],
                "retrieved_context": []
            }

        # ✅ Step 3: Build prompt
        prompt = build_prompt(query, docs)

        # ✅ Step 4: Call LLM
        response = call_bedrock_llm(prompt)

        # ✅ STRICT no-context handling
        if "i don't have enough information" in response.lower():
            log_interaction(query, response, "NO_CONTEXT", None)

            return {
                "response": response,
                "sources": [],  # ✅ NO SOURCES
                "highlight_text": response[:200],
                "retrieved_context": docs
            }

        # ✅ Step 5: Source selection (STRICT TOP-1)
        sources = []

        if docs:
            best_doc = docs[0]  # ✅ always best after sorting

            metadata = best_doc.get("metadata", {})

            sources.append({
                "name": metadata.get("filename", "Unknown"),
                "type": metadata.get("source_type", "unknown"),
                "url": metadata.get("source_url"),
                "retrieved_context": docs
            })

        # ✅ Fallback: use best ranked doc (top-1)
        if not sources and docs:
            best_doc = docs[0]
            metadata = best_doc.get("metadata", {})
            sources.append({
                "name": metadata.get("filename", "Unknown"),
                "type": metadata.get("source_type", "unknown"),
                "url": metadata.get("source_url")
            })

        # ✅ Remove duplicates
        sources = list({s["name"]: s for s in sources}.values())

        log_interaction(query, response, "RAG", None)
        is_blocked, fallback = check_output_guardrail(response)

        if is_blocked:
            # Replace the unsafe response with the safe fallback message.
            # Sources are cleared so the user cannot infer what was retrieved.
            return {"response": fallback, "sources": []}

        return {
            "response": response,
            "sources": sources,
            "highlight_text": response[:200],
            "retrieved_context": docs,
            # "grounding_score": grounding_score,
            # "retrieval_time_ms": retrieval_time_ms,
            # "llm_time_ms": llm_time_ms,
        }

    except Exception:
        error_msg = "❌ Error in chatbot"
        logger.exception(error_msg)
        log_interaction(query, error_msg, "ERROR", None)

        return {
            "response": "Something went wrong while processing your request.",
            "sources": [],
            "retrieved_context": []
        }
