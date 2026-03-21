# app/rag.py

import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR


def get_vectordb():
    """
    Shared vector DB instance (used by main.py for delete/list APIs)
    """
    try:
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )

        print("✅ Vector DB initialized")

        return vectordb

    except Exception as e:
        print(f"❌ Vector DB Error: {str(e)}")
        raise