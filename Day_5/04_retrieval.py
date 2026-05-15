"""
04_retrieval.py
Retrieval example using FAISS and LangChain
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Example chunks
documents = ["RAG uses retrieval to find relevant information.",
             "LangChain supports various retrievers."]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

query = "What does RAG use?"
results = retriever.get_relevant_documents(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
