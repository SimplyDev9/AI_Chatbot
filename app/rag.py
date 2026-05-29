# app/rag.py

import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR
from app.logger import logger


def get_vectordb():
    """
    Shared vector DB instance (used by main.py for delete/list APIs)
    """
    try:
        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            # model_kwargs={
            #     "dimensions": 512,
            #     "normalize": True
            # }
        )

        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )

        logger.info("Vector DB initialized")

        return vectordb

    except Exception:
        logger.exception("Vector DB initialization error")
        raise