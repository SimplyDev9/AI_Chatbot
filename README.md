# 🤖 AI Chatbot with RAG (Retrieval-Augmented Generation)

A production-ready **AI chatbot** built using a **Retrieval-Augmented Generation (RAG)** pipeline.
It ingests documents, stores embeddings in a vector database, and retrieves relevant context to generate intelligent responses.

---

## 🚀 Features

* 📂 **Document Ingestion**

  * Supports `.txt`, `.pdf`, `.docx`, `.pptx`
  * Automatic text extraction and chunking

* 🧠 **RAG Pipeline**

  * Context-aware responses using retrieved documents
  * Efficient semantic search with embeddings

* 🗄️ **Vector Database**

  * Powered by **ChromaDB**
  * Persistent storage of embeddings

* 🔄 **Incremental Updates**

  * Detects file changes using hashing
  * Avoids reprocessing unchanged files

* ☁️ **AWS Bedrock Integration**

  * Uses Titan embeddings (or configurable models)

* 📝 **Logging System**

  * Tracks ingestion and processing steps

---

## 🏗️ Project Structure

```
AI_Chatbot/
│
├── app/
│   ├── main.py                # Entry point
│   ├── config.py              # Configuration (paths, env, models)
│   ├── ingest_corpus.py       # Data ingestion pipeline
│   ├── retriever.py           # Retrieval logic
│   ├── rag.py                 # RAG pipeline
│   ├── llm.py                 # LLM integration
│   ├── document_loader.py     # File loaders
│   ├── prompt_builder.py      # Prompt engineering
│   ├── logger.py              # Logging utility
│
├── data/
│   ├── corpus/                # Input documents
│   ├── chroma_db/             # Vector database
│
├── logs/                      # Application logs
├── .env                       # Environment variables
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/SimplyDev9/AI_Chatbot.git
cd AI_Chatbot
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure environment variables

Create a `.env` file:

```env
ENV=dev

DATA_DIR=./data
CORPUS_DIR=./data/corpus
CHROMA_DIR=./data/chroma_db

LOG_DIR=./logs

AWS_DEFAULT_REGION=your-region
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

---

## 📥 Ingest Documents

Place your files inside:

```
data/corpus/
```

Then run:

```bash
python -m app.ingest_corpus
```

---

## 💬 Run the Chatbot

```bash
python -m app.main
```

---

## 🧠 How It Works

1. Documents are loaded and parsed
2. Text is split into chunks
3. Embeddings are generated using AWS Bedrock
4. Stored in ChromaDB
5. On query:

   * Relevant chunks are retrieved
   * Passed to LLM for response generation

---

## 🔐 Security Notes

* Never commit `.env` file
* Keep AWS credentials secure
* Use IAM roles in production

---

## 🧪 Future Improvements

* ✅ API layer (FastAPI)
* ✅ Web UI (React / Streamlit)
* ✅ Docker support
* ⏳ Multi-user support
* ⏳ Streaming responses

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* LangChain
* ChromaDB
* AWS Bedrock
* Open-source community

---
