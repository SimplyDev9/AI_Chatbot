# app/retriever.py

from app.knowledge_base import KnowledgeBase

kb = KnowledgeBase()


def retrieve_documents(query: str, k: int = 5):
    try:
        print(f"🔍 Retrieving documents for: {query}")

        docs = kb.search(query, k=2)

        if not docs:
            print("⚠️ No documents found")
            return []

        print(f"📄 Retrieved {len(docs)} docs")

        return docs[:k]

    except Exception as e:
        print(f"❌ Retriever error: {str(e)}")
        return []