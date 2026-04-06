# app/llm.py

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from app.logger import logger

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

try:
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        model_kwargs={
            "temperature": 0,
            "top_p": 1
        }
    )
    logger.info("LLM initialized successfully")

except Exception as e:
    logger.exception("LLM Initialization Error")
    llm = None


def call_bedrock_llm(prompt: str) -> str:
    """
    Main LLM call for RAG system
    """
    try:
        if not llm:
            raise Exception("LLM not initialized")

        logger.debug("Calling Bedrock LLM...")

        response = llm.invoke(prompt)

        output = getattr(response, "content", str(response))

        logger.debug("LLM response received")

        return output

    except Exception:
        logger.exception("LLM Error during call")
        return "Error generating response from model."


# Optional (can keep for future flexibility)
def call_llm(user_query: str, context: str = None) -> str:
    try:
        prompt = user_query

        if context:
            prompt = f"Context:\n{context}\n\nUser Query:\n{user_query}"

        return call_bedrock_llm(prompt)

    except Exception:
        logger.exception("call_llm Error")
        return ""