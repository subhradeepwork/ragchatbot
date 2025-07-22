from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain
