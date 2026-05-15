"""
02_langchain_pipeline.py
Basic RAG pipeline using LangChain
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
documents = ["LangChain is a framework for developing applications powered by language models.",
             "RAG combines retrieval and generation for better answers."]

# Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.create_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Setup retriever
retriever = vectorstore.as_retriever()

# Setup QA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Example query
query = "What is LangChain?"
result = qa.run(query)
print("Q:", query)
print("A:", result)
