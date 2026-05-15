"""
05_mini_rag_system.py
Build a mini RAG system using LangChain
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Example documents
documents = [
    "LangChain enables RAG pipelines easily.",
    "Retrieval-Augmented Generation improves LLM answers.",
    "Chunking helps in efficient retrieval.",
    "FAISS is a fast vector store for similarity search."
]

# Split documents
splitter = CharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.create_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Setup RAG pipeline
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# User query
query = "How does RAG improve LLMs?"
result = qa.run(query)
print("Q:", query)
print("A:", result)
