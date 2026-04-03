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

            # 🔥 Collect ALL results (NO threshold filtering)
            for doc, score in results:
                print(f"🔎 Score: {score} | Text preview: {doc.page_content[:50]}")

                docs.append({
                    "text": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"), # ✅ for citations
                    "source_type": doc.metadata.get("source_type", "unknown"),
                    "source_url": doc.metadata.get("source_url")
                })

            if not docs:
                print("⚠️ No documents retrieved")
                return []

            # 🔥 Sort by best score (lower = better)
            docs = sorted(docs, key=lambda x: x["score"])

            # 🔥 Take top-k (you can tune this later)
            docs = docs[:3]

            print(f"✅ Selected docs count: {len(docs)}")

            return docs

        except Exception as e:
            print(f"❌ KB Search Error: {str(e)}")
            return []

    def file_exists_in_db(vectordb, filename):
        try:
            results = vectordb.get(where={"filename": filename})
            return len(results.get("ids", [])) > 0
        except Exception as e:
            print(f"❌ DB check error: {str(e)}")
            return False