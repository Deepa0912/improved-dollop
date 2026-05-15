"""
03_document_loading.py
Document loading and chunking example
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load a text file (replace 'sample.txt' with your file)
# loader = TextLoader('sample.txt')
# documents = loader.load()
documents = ["This is a sample document for RAG demonstration. It will be split into chunks."]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.create_documents(documents)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")
