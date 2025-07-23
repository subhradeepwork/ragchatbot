
import os
import re
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def embed_and_store_pdf(pdf_path: str) -> bool:
    try:
        print(f"[INFO] Loading PDF from: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[INFO] Loaded {len(documents)} pages.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        print(f"[INFO] FAISS saved to: {INDEX_PATH}")
        return True
    except Exception as e:
        print(f"[EXCEPTION in embed_and_store_pdf] {e}")
        return False

def query_rag(question: str):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOpenAI(temperature=0)

        if "summary" in question.lower() or "summarize" in question.lower():
            docs = retriever.get_relevant_documents("summarize this document")
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)
            sources = format_sources(docs)
            return summary.strip(), sources

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain({"query": question})
        docs = result.get("source_documents", [])
        sources = format_sources(docs)
        return result["result"].strip(), sources

    except Exception as e:
        print(f"[EXCEPTION in query_rag] {e}")
        return "An error occurred while answering the question.", []

def smart_truncate(text, limit=400):
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) <= limit:
        return text
    cutoff = text[:limit].rfind('.')
    if cutoff == -1 or cutoff < 100:
        return text[:limit] + "..."
    return text[:cutoff + 1] + "..."

def format_sources(docs):
    sources = []
    for i, doc in enumerate(docs[:3]):
        page = doc.metadata.get("page", "?")
        cleaned_text = doc.page_content.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\f", " ")
        snippet = smart_truncate(cleaned_text)
        sources.append(f"📄 Page {page} — {snippet}")
    return sources
