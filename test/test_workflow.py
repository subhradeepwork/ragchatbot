import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pdf_processor import extract_text
from core.chunker import chunk_text
from core.embedder import get_embedder
from core.vector_store import build_vector_store, save_vector_store
from core.rag_pipeline import get_rag_chain

if __name__ == "__main__":
    text = extract_text("data/legal_case.pdf")
    print(f"Extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Chunked into {len(chunks)} segments")

    embedder = get_embedder()
    vector_store = build_vector_store(chunks, embedder)
    save_vector_store(vector_store)

    rag_chain = get_rag_chain(vector_store)

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = rag_chain({"query": query})
        print("Answer:", result["result"])
        print("--- Source Chunks ---")
        for doc in result["source_documents"]:
            print(doc.page_content)
            print("---")
