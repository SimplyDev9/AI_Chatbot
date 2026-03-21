# app/llm.py

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

try:
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        model_kwargs={"max_tokens": 2000}
    )
    print("✅ LLM initialized successfully")

except Exception as e:
    print(f"❌ LLM Initialization Error: {str(e)}")
    llm = None


def call_bedrock_llm(prompt: str) -> str:
    """
    Main LLM call for RAG system
    """
    try:
        if not llm:
            raise Exception("LLM not initialized")

        print("🤖 Calling Bedrock LLM...")

        response = llm.invoke(prompt)

        output = getattr(response, "content", str(response))

        print("✅ LLM response received")

        return output

    except Exception as e:
        print(f"❌ LLM Error: {str(e)}")
        return "Error generating response from model."


# Optional (can keep for future flexibility)
def call_llm(user_query: str, context: str = None) -> str:
    try:
        prompt = user_query

        if context:
            prompt = f"Context:\n{context}\n\nUser Query:\n{user_query}"

        return call_bedrock_llm(prompt)

    except Exception as e:
        print(f"❌ call_llm Error: {str(e)}")
        return ""