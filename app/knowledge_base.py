# app/knowledge_base.py

import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR


class KnowledgeBase:

    def __init__(self):
        try:
            self.embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                region_name=os.getenv("AWS_DEFAULT_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            self.vectordb = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings
            )

            print("✅ KnowledgeBase initialized")

        except Exception as e:
            print(f"❌ KB Init Error: {str(e)}")
            raise

    def search(self, query: str, k: int = 10):
        try:
            print(f"🔍 Searching KB: {query}")

            results = self.vectordb.similarity_search_with_score(query, k=k)

            docs = []

            for doc, score in results:
                docs.append({
                    "text": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })

            print(f"📄 Found {len(docs)} documents")

            return docs

        except Exception as e:
            print(f"❌ KB Search Error: {str(e)}")
            return []