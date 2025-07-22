
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def embed_and_store_pdf(pdf_path: str) -> bool:
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        return True
    except Exception as e:
        print(f"Error embedding PDF: {e}")
        return False

def query_rag(question: str):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa_chain({"query": question})

        # Get source documents
        docs = retriever.get_relevant_documents(question)
        sources = [doc.page_content[:300] + "..." for doc in docs]  # truncate for display
        return result["result"], sources
    except Exception as e:
        print(f"Error querying RAG: {e}")
        return "An error occurred while answering the question.", []
