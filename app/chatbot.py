# app/chatbot.py

from app.llm import call_bedrock_llm
from app.logger import log_interaction
from app.retriever import retrieve_documents
from app.prompt_builder import build_prompt


def answer_query(query: str):
    try:
        print(f"🔥 Incoming query: {query}")

        docs = retrieve_documents(query)

        if not docs:
            response = "I don't have enough information in the knowledge base to answer this."
            log_interaction(query, response, "NO_CONTEXT", None)
            return response

        prompt = build_prompt(query, docs)

        response = call_bedrock_llm(prompt)

        log_interaction(query, response, "RAG", None)

        return response

    except Exception as e:
        error_msg = f"❌ Error in chatbot: {str(e)}"
        print(error_msg)
        log_interaction(query, error_msg, "ERROR", None)
        return "Something went wrong while processing your request."