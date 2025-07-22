from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def build_vector_store(chunks, embedder):
    docs = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
    vector_store = FAISS.from_documents(docs, embedder)
    return vector_store

def save_vector_store(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vector_store(embedder, path="faiss_index"):
    return FAISS.load_local(path, embedder)
