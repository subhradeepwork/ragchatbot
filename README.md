
# RAG-Powered Legal Document Chatbot

This project is a Phase 2 implementation of a **Retrieval-Augmented Generation (RAG)** chatbot system. It allows users to:

-  Upload legal PDF documents
-  Ask natural language questions about their content
-  Receive accurate answers along with source citations
-  Uses LangChain + OpenAI for semantic understanding
-  Built with FastAPI (backend) and Gradio (frontend)

---

## 📁 Project Structure

```
ragchatbot/
├── backend/              # FastAPI app with /upload and /chat
│   └── main.py
├── core/                 # Core logic for RAG processing
│   └── rag_engine.py
├── frontend/             # Gradio app UI
│   └── app.py
├── data/uploads/         # Stores uploaded PDF files
├── faiss_index/          # Persistent FAISS vector database
├── test/                 # CLI test scripts (optional)
├── .env                  # API keys (OpenAI)
├── requirements.txt      # Project dependencies
└── README.md
```

---

## 🔄 Application Flow

### 📌 1. File Upload
- User uploads a `.pdf` via the Gradio UI.
- This triggers a POST to `POST /upload` on the FastAPI backend.
- Backend stores the file and passes it to `embed_and_store_pdf()`.

### 📌 2. Embedding Logic (rag_engine.py)
- PDF is loaded and chunked using LangChain.
- Embeddings are generated with OpenAIEmbeddings.
- Chunks + embeddings are stored in FAISS (`faiss_index/`).

### 📌 3. Asking a Question
- User types a question into the Gradio chatbot.
- It calls `POST /chat` → `query_rag(question)`.
- Retrieves top-k chunks from FAISS, sends them to OpenAI (ChatModel).

### 📌 4. Answer Generation
- If the question includes "summary", `map_reduce` summarization is triggered.
- Otherwise, a `RetrievalQA` chain is used.
- The response is returned along with 3 smart-truncated source snippets.

---

## ⚙️ Script Flow & Invocation

| Script         | Called By             | Purpose |
|----------------|-----------------------|---------|
| `main.py`      | FastAPI entry point   | API for /upload and /chat |
| `rag_engine.py`| `main.py`             | Core logic (embed, retrieve, generate) |
| `app.py`       | Standalone via `python frontend/app.py` | Gradio UI for chat & upload |
| `test_workflow.py` | CLI (optional)    | Manual testing for backend functions |

---

##  RAG Details

- **LLM**: `ChatOpenAI` from LangChain
- **Embeddings**: `OpenAIEmbeddings` (can be swapped)
- **Vector Store**: FAISS (disk-persisted)
- **Chunking**: RecursiveCharacterTextSplitter
- **Summarization**: LangChain `load_summarize_chain` with `map_reduce`
- **Sources**: Smart-truncated with full sentence endings

---

##  Running Locally

### ✅ Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ Step 2: Set up `.env`

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
```

### ✅ Step 3: Start backend

```bash
cd backend
uvicorn main:app --reload
```

### ✅ Step 4: Start frontend (Gradio)

```bash
cd frontend
python app.py
```

---

## ✅ Current Features

- Upload a PDF and process it into a FAISS DB
- Ask questions and get answers with sources
- Smart source truncation (no mid-sentence cutoffs)
- Summary vs QA logic switching

---

## 🔜 Next Phase (Planned)

-  Save chat history per session
-  Multi-PDF support
-  PDF viewer + jump-to-page
-  Local LLM (Ollama, GPT4All)
-  User authentication

---

##  Author Notes

This system was built with real-time debugging and production-readiness in mind.  
You're free to fork, extend, or contribute!

---

