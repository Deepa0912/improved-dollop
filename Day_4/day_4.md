# Day 4 – Sentence Transformers, Embeddings & Vector Search

Embeddings are the backbone of modern semantic AI applications — from semantic search and RAG pipelines to recommendation systems and duplicate detection. Today we explore **Sentence Transformers** (a free, open-source library), how to generate high-quality text embeddings, measure similarity using **cosine similarity**, and perform blazing-fast nearest-neighbour search with **FAISS** — all without any paid API.

---

## 1. What Are Embeddings?

An **embedding** is a dense vector of floating-point numbers that encodes the *meaning* of a piece of text in a high-dimensional space.

```
"The cat sat on the mat"  →  [0.12, -0.87, 0.34, 0.91, ...]   (768 numbers)
"A feline rested on a rug"  →  [0.11, -0.85, 0.36, 0.89, ...]   (similar!)
"The stock market crashed"  →  [-0.52, 0.21, -0.67, 0.03, ...]  (very different)
```

### Why Embeddings Matter

| Property | Explanation |
|---|---|
| **Semantic similarity** | Texts with similar meaning have vectors close together |
| **Language-agnostic** | Multilingual models map "cat" and "chat" (French) near each other |
| **Fixed-size** | No matter how long the input, output is always `d` floats |
| **Composable** | Average embeddings, do arithmetic, cluster, search |

### Embedding Dimensions in Practice

| Model | Embedding Dim | Use Case |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast, general purpose |
| `all-mpnet-base-v2` | 768 | High quality, general purpose |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ languages |
| `text-embedding-3-small` (OpenAI) | 1536 | Cloud, paid |
| `text-embedding-ada-002` (OpenAI) | 1536 | Cloud, paid (legacy) |

---

## 2. Sentence Transformers Library

[`sentence-transformers`](https://www.sbert.net/) is a Python library built on top of HuggingFace Transformers that makes generating **sentence-level embeddings** simple.

### Installation

```bash
pip install sentence-transformers
```

### How It Works

Standard BERT/transformer models output a **token-level** embedding tensor of shape `(batch, seq_len, hidden_dim)`. Sentence Transformers applies a **pooling layer** (usually mean pooling over the token embeddings) to collapse this into a single **sentence vector** of shape `(hidden_dim,)`.

```
Text → Tokenizer → Transformer → Token Embeddings (seq_len × 768)
                                        ↓ Mean Pooling
                              Sentence Embedding (768,)
```

### Quick Start

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained model (downloads ~90MB on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# Encode sentences → numpy array of shape (3, 384)
embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")          # (3, 384)
print(f"Type : {type(embeddings)}")          # <class 'numpy.ndarray'>
print(f"First 5 values: {embeddings[0][:5]}")
```

### Batch Encoding for Performance

```python
# Large-scale encoding — process in batches
embeddings = model.encode(
    sentences,
    batch_size=64,           # Process 64 sentences at a time
    show_progress_bar=True,  # Show tqdm progress bar
    normalize_embeddings=True,  # L2-normalise (unit vectors)
    convert_to_numpy=True,   # Default: numpy array
    # convert_to_tensor=True,  # Alternatively: PyTorch tensor
)
```

> **Tip:** Set `normalize_embeddings=True` when you will use **cosine similarity** — it converts dot product to cosine similarity automatically.

---

## 3. Cosine Similarity

**Cosine similarity** measures the angle between two vectors. It is the standard metric for comparing embeddings because it is **magnitude-independent** — only the *direction* (meaning) matters, not the *length*.

### Formula

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

- Range: **−1** (opposite) → **0** (orthogonal/unrelated) → **+1** (identical)
- For semantic similarity, scores above **0.7** are typically "similar"

### Computing Cosine Similarity

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning.",
    "Deep learning is a subset of ML.",
    "The pizza was delicious.",
]

embs = model.encode(sentences, normalize_embeddings=True)

# Pairwise cosine similarity matrix (3×3)
sim_matrix = cosine_similarity(embs)
print(sim_matrix.round(3))
# Expected: sentences 0 & 1 have high similarity, sentence 2 is low
```

### Using `sentence_transformers` Built-in Utility

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

query   = "What is natural language processing?"
answers = [
    "NLP is a field of AI that deals with understanding text.",
    "The Eiffel Tower is located in Paris.",
    "Machine learning uses statistical methods.",
]

query_emb   = model.encode(query, convert_to_tensor=True)
answer_embs = model.encode(answers, convert_to_tensor=True)

scores = util.cos_sim(query_emb, answer_embs)  # Shape: (1, 3)
print(scores)

# Rank answers by similarity
for i, score in enumerate(scores[0]):
    print(f"  [{score:.4f}] {answers[i]}")
```

---

## 4. FAISS – Fast Vector Search

**FAISS** (Facebook AI Similarity Search) is a library for **efficient similarity search** and **clustering** of dense vectors. It can search millions of vectors in milliseconds on a CPU — and even faster on GPU.

### Why FAISS?

| Challenge | FAISS Solution |
|---|---|
| Exact search over 1M vectors is slow (O(n)) | Approximate Nearest Neighbour (ANN) indices |
| Memory-intensive at scale | Product Quantization (PQ) compression |
| GPU acceleration needed | Native CUDA support (`faiss-gpu`) |

### Installation

```bash
# CPU version (free, works on all platforms)
pip install faiss-cpu

# GPU version (requires CUDA)
# pip install faiss-gpu
```

### Core FAISS Concepts

| Term | Explanation |
|---|---|
| **Index** | A data structure that stores vectors and supports search |
| **`d`** | Vector dimensionality (must match your embedding model) |
| **`k`** | Number of nearest neighbours to retrieve |
| **`IndexFlatL2`** | Exact L2 (Euclidean) distance — brute force, most accurate |
| **`IndexFlatIP`** | Exact Inner Product (= cosine sim for L2-normalised vectors) |
| **`IndexIVFFlat`** | Inverted File — partitions space into `nlist` clusters (faster, approximate) |
| **`IndexHNSWFlat`** | Hierarchical Navigable Small World — graph-based ANN |

### Basic FAISS Workflow

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "Python is a versatile programming language.",
    "Machine learning involves training models on data.",
    "The Eiffel Tower is in Paris, France.",
    "Deep learning uses neural networks.",
    "Neural networks are inspired by the brain.",
    "Paris is the capital of France.",
    "Python supports object-oriented programming.",
]

doc_embeddings = model.encode(documents, normalize_embeddings=True)
d = doc_embeddings.shape[1]  # 384

# 2. Build an exact Inner Product index
index = faiss.IndexFlatIP(d)  # Inner Product ≡ Cosine sim (after L2 norm)

# 3. Add vectors to the index
index.add(doc_embeddings.astype("float32"))
print(f"Index contains {index.ntotal} vectors")

# 4. Query
query = "What programming languages are popular for AI?"
query_emb = model.encode([query], normalize_embeddings=True).astype("float32")

k = 3  # Top-3 results
distances, indices = index.search(query_emb, k)

print(f"\nTop-{k} results for: '{query}'")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {rank+1}. [{dist:.4f}] {documents[idx]}")
```

### Saving and Loading a FAISS Index

```python
# Save to disk
faiss.write_index(index, "my_index.faiss")

# Load from disk
index = faiss.read_index("my_index.faiss")
```

### Scalable IVF Index (for large datasets)

```python
# For large corpora (10k+ docs), use IVFFlat for speed
nlist = 50   # Number of Voronoi cells (clusters)
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# IVF index must be trained before adding vectors
index_ivf.train(doc_embeddings.astype("float32"))
index_ivf.add(doc_embeddings.astype("float32"))

# nprobe: how many clusters to search (accuracy vs speed tradeoff)
index_ivf.nprobe = 10

distances, indices = index_ivf.search(query_emb, k)
```

---

## 5. Metadata Filtering with FAISS

FAISS only stores vectors — it has **no built-in metadata support**. The standard pattern is to maintain a **parallel Python list or dictionary** of metadata keyed by the same integer ID that FAISS uses.

### Pattern: Parallel Metadata Store

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents with rich metadata
corpus = [
    {"text": "Python is great for data science.",      "category": "programming", "year": 2023},
    {"text": "Neural networks learn from examples.",   "category": "ml",          "year": 2022},
    {"text": "The Louvre is a famous museum in Paris.","category": "travel",      "year": 2021},
    {"text": "PyTorch is a deep learning framework.",  "category": "ml",          "year": 2023},
    {"text": "Rome has ancient Roman architecture.",   "category": "travel",      "year": 2022},
    {"text": "Scikit-learn is great for classical ML.","category": "ml",          "year": 2021},
    {"text": "JavaScript is used for web development.","category": "programming", "year": 2023},
]

texts      = [doc["text"] for doc in corpus]
embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
d          = embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# ── Metadata Filtering (Post-filter pattern) ──────────────────────────────────
def search_with_filter(query: str, k: int = 5, category: str = None, year: int = None):
    """Search FAISS then filter results by metadata."""
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    # Retrieve more candidates than needed to allow filtering
    candidates = min(index.ntotal, k * 4)
    distances, indices = index.search(query_emb, candidates)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        doc = corpus[idx]
        # Apply metadata filters
        if category and doc["category"] != category:
            continue
        if year and doc["year"] != year:
            continue
        results.append({"score": float(dist), **doc})
        if len(results) >= k:
            break

    return results


# Example queries with filters
print("── Query: AI frameworks (category=ml) ──")
results = search_with_filter("best frameworks for machine learning", k=3, category="ml")
for r in results:
    print(f"  [{r['score']:.4f}] ({r['year']}) {r['text']}")

print("\n── Query: European destinations (year=2022) ──")
results = search_with_filter("places to visit in Europe", k=3, year=2022)
for r in results:
    print(f"  [{r['score']:.4f}] ({r['year']}) {r['text']}")
```

---

## 6. Vector Database Comparison: FAISS vs ChromaDB vs Pinecone

When building production RAG systems, you need to choose the right vector store. Here's an overview:

| Feature | **FAISS** | **ChromaDB** | **Pinecone** |
|---|---|---|---|
| **Type** | Library | Embedded / Server DB | Managed Cloud Service |
| **Cost** | Free (open-source) | Free (open-source) | Free tier + paid plans |
| **Metadata filtering** | Manual (parallel dict) | ✅ Built-in | ✅ Built-in |
| **Persistence** | Manual (write_index) | ✅ Automatic | ✅ Automatic |
| **Scalability** | Single machine | Single machine / server | Multi-billion vectors |
| **Setup complexity** | Low | Very Low | Low (API key) |
| **Search accuracy** | Exact or ANN | Exact (HNSW) | ANN (managed) |
| **GPU support** | ✅ faiss-gpu | ❌ | N/A (cloud) |
| **Best for** | Research, custom pipelines | Local dev, prototypes | Production at scale |

### ChromaDB Quick Example

```python
# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()  # In-memory; use PersistentClient for disk

# ChromaDB can use Sentence Transformers directly
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.create_collection(
    name="my_docs",
    embedding_function=ef,
)

# Add documents with metadata
collection.add(
    documents=[
        "Python is widely used in data science.",
        "The Colosseum is in Rome, Italy.",
        "PyTorch enables GPU-accelerated deep learning.",
    ],
    metadatas=[
        {"category": "programming"},
        {"category": "travel"},
        {"category": "ml"},
    ],
    ids=["doc1", "doc2", "doc3"],
)

# Query with metadata filter
results = collection.query(
    query_texts=["machine learning frameworks"],
    n_results=2,
    where={"category": "ml"},  # Built-in metadata filter!
)
print(results["documents"])
```

### Decision Guide

```
Need a quick prototype or research?     → FAISS
Need metadata filters + local storage?  → ChromaDB
Need production scale (>10M vectors)?   → Pinecone
Need full control + no cloud?           → FAISS or ChromaDB
```

---

## 7. Semantic Search Pipeline (End-to-End)

Putting it all together — a real semantic search engine:

```python
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class SemanticSearchEngine:
    """
    Lightweight semantic search engine using Sentence Transformers + FAISS.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model    = SentenceTransformer(model_name)
        self.index    = None
        self.metadata = []   # Parallel metadata store
        self.d        = None

    def build_index(self, documents: list[dict], text_key: str = "text"):
        """Encode documents and build a FAISS index."""
        texts = [doc[text_key] for doc in documents]
        self.metadata = documents

        embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).astype("float32")

        self.d     = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(embs)
        print(f"Index built: {self.index.ntotal} vectors of dim {self.d}")

    def search(self, query: str, k: int = 5, filters: dict = None) -> list[dict]:
        """Search index, optionally filtering by metadata fields."""
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        candidates = min(self.index.ntotal, k * 6)
        distances, indices = self.index.search(query_emb, candidates)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc = self.metadata[idx]
            if filters:
                if not all(doc.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append({"score": round(float(dist), 4), **doc})
            if len(results) >= k:
                break
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_meta.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved to {path}.faiss + {path}_meta.json")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_meta.json") as f:
            self.metadata = json.load(f)
        self.d = self.index.d
        print(f"Loaded index: {self.index.ntotal} vectors")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        {"text": "FAISS enables fast vector similarity search.",      "topic": "databases", "lang": "en"},
        {"text": "Sentence Transformers generate sentence embeddings.", "topic": "nlp",      "lang": "en"},
        {"text": "PyTorch is a popular deep learning framework.",       "topic": "ml",       "lang": "en"},
        {"text": "Cosine similarity measures angle between vectors.",   "topic": "math",     "lang": "en"},
        {"text": "ChromaDB is an open-source vector database.",        "topic": "databases", "lang": "en"},
        {"text": "Pinecone is a managed vector database service.",      "topic": "databases", "lang": "en"},
        {"text": "BERT is a powerful NLP model from Google.",          "topic": "nlp",       "lang": "en"},
        {"text": "RAG combines retrieval with language generation.",    "topic": "nlp",       "lang": "en"},
    ]

    engine = SemanticSearchEngine()
    engine.build_index(docs)

    print("\n── Semantic Search: 'vector databases for AI' ──")
    for r in engine.search("vector databases for AI", k=3):
        print(f"  [{r['score']:.4f}] [{r['topic']}] {r['text']}")

    print("\n── With filter topic=nlp ──")
    for r in engine.search("language models and transformers", k=3, filters={"topic": "nlp"}):
        print(f"  [{r['score']:.4f}] [{r['topic']}] {r['text']}")
```

---

## 8. Hands-On Scripts

The following Python files are provided in `Day_4/`:

| File | What It Demonstrates |
|---|---|
| `01_embeddings_basics.py` | Generate embeddings with Sentence Transformers, inspect shapes, visualise similarity |
| `02_cosine_similarity.py` | Pairwise similarity matrix, ranking, semantic textual similarity (STS) |
| `03_faiss_search.py` | Build Flat and IVF FAISS indices, exact vs approximate search |
| `04_metadata_filtering.py` | Parallel metadata store, post-filter pattern with FAISS |
| `05_semantic_engine.py` | Full end-to-end semantic search engine class with save/load |

Run any script:
```bash
cd Day_4
python 01_embeddings_basics.py
python 02_cosine_similarity.py
python 03_faiss_search.py
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Embeddings** | Dense vectors encoding text meaning; similar texts have similar vectors |
| **Sentence Transformers** | Free library for high-quality sentence embeddings; `model.encode()` is all you need |
| **Cosine Similarity** | Angle between vectors; range −1 to +1; use `util.cos_sim()` or sklearn |
| **FAISS** | Ultra-fast similarity search library; `IndexFlatIP` for exact, `IndexIVFFlat` for ANN |
| **Metadata Filtering** | FAISS has no metadata — maintain a parallel Python list; ChromaDB has it built-in |
| **FAISS vs ChromaDB** | FAISS = raw speed & flexibility; ChromaDB = batteries-included local dev |
| **Pinecone** | Managed cloud solution for production scale; paid beyond free tier |
| **Semantic Search** | Query → embed → FAISS search → rank by cosine score → return top-k |

---

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [FAISS Wiki – Indexing Overview](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [MTEB Leaderboard – Best Embedding Models](https://huggingface.co/spaces/mteb/leaderboard)
