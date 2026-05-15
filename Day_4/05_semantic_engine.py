"""
Day 4 – Script 05: Full Semantic Search Engine (End-to-End)
===========================================================
A reusable SemanticSearchEngine class that wraps:
  - Sentence Transformers (encoding)
  - FAISS IndexFlatIP (search)
  - Parallel metadata store
  - Save / Load (index + metadata)
  - Metadata filtering (AND logic)
  - Interactive REPL demo

Run:
    python 05_semantic_engine.py

Requirements:
    pip install sentence-transformers faiss-cpu
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════════════════════
# SemanticSearchEngine class
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSearchEngine:
    """
    Lightweight semantic search engine backed by Sentence Transformers + FAISS.

    Usage:
        engine = SemanticSearchEngine()
        engine.build_index(documents)          # list of dicts with "text" key
        results = engine.search("my query", k=5, filters={"category": "ai"})
        engine.save("my_engine")
        engine.load("my_engine")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}...")
        self.model    = SentenceTransformer(model_name)
        self.index    = None
        self.metadata: list[dict] = []
        self.d        = None
        print("Model ready ✓")

    # ─────────────────────────────────────────────────────────────────────────
    def build_index(
        self,
        documents: list[dict],
        text_key: str = "text",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """Encode all documents and build a FAISS IndexFlatIP."""
        if not documents:
            raise ValueError("documents list is empty")

        texts          = [doc[text_key] for doc in documents]
        self.metadata  = list(documents)

        print(f"Encoding {len(texts)} documents...")
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype("float32")

        self.d     = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(embs)
        print(f"Index built: {self.index.ntotal} vectors, dim={self.d}")

    # ─────────────────────────────────────────────────────────────────────────
    def add_documents(self, documents: list[dict], text_key: str = "text") -> None:
        """Incrementally add more documents to an existing index."""
        if self.index is None:
            raise RuntimeError("Call build_index() first.")
        texts = [doc[text_key] for doc in documents]
        embs  = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index.add(embs)
        self.metadata.extend(documents)
        print(f"Added {len(documents)} docs. Total: {self.index.ntotal}")

    # ─────────────────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
        over_fetch_factor: int = 6,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filtering.

        Args:
            query              : Natural language query string
            k                  : Number of results to return
            filters            : Dict of {field: value} to apply (AND logic)
            over_fetch_factor  : Multiply k by this to allow for filtering

        Returns:
            List of result dicts with 'score' prepended to the metadata.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_emb  = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        candidates = min(self.index.ntotal, k * over_fetch_factor)
        distances, indices = self.index.search(query_emb, candidates)

        results: list[dict] = []
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

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path_prefix: str) -> None:
        """Save FAISS index (.faiss) and metadata (.json) to disk."""
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        meta_path = f"{path_prefix}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"d": self.d, "documents": self.metadata}, f, indent=2, ensure_ascii=False)
        print(f"Saved → {path_prefix}.faiss  +  {meta_path}")

    # ─────────────────────────────────────────────────────────────────────────
    def load(self, path_prefix: str) -> None:
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(f"{path_prefix}.faiss")
        meta_path  = f"{path_prefix}_meta.json"
        with open(meta_path, encoding="utf-8") as f:
            payload = json.load(f)
        self.d        = payload["d"]
        self.metadata = payload["documents"]
        print(f"Loaded ← {path_prefix}.faiss  ({self.index.ntotal} vectors, dim={self.d})")

    # ─────────────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Return basic statistics about the index."""
        categories = {}
        for doc in self.metadata:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_documents": len(self.metadata),
            "embedding_dim"  : self.d,
            "by_category"    : categories,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Demo corpus
# ═══════════════════════════════════════════════════════════════════════════════

CORPUS = [
    # AI / Embeddings
    {"text": "FAISS is a library for efficient similarity search of dense vectors.", "category": "ai",          "difficulty": "intermediate"},
    {"text": "Sentence Transformers encode sentences into fixed-size dense vectors.", "category": "ai",          "difficulty": "beginner"},
    {"text": "Cosine similarity measures the angle between two embedding vectors.",   "category": "ai",          "difficulty": "beginner"},
    {"text": "Vector databases like Pinecone and ChromaDB store embeddings at scale.","category": "ai",          "difficulty": "intermediate"},
    {"text": "RAG pipelines retrieve relevant documents before generating answers.",   "category": "ai",          "difficulty": "advanced"},
    {"text": "BERT generates contextualised word embeddings using transformers.",      "category": "ai",          "difficulty": "intermediate"},
    # Programming
    {"text": "Python list comprehensions offer a concise way to create lists.",       "category": "programming", "difficulty": "beginner"},
    {"text": "Async/await in Python enables non-blocking IO operations.",             "category": "programming", "difficulty": "intermediate"},
    {"text": "Docker containers package applications with all dependencies.",          "category": "programming", "difficulty": "intermediate"},
    {"text": "REST APIs use HTTP methods: GET, POST, PUT, DELETE.",                   "category": "programming", "difficulty": "beginner"},
    # Science
    {"text": "Photosynthesis converts sunlight and CO₂ into glucose and oxygen.",     "category": "science",     "difficulty": "beginner"},
    {"text": "Quantum entanglement links particles so their states are correlated.",   "category": "science",     "difficulty": "advanced"},
    {"text": "DNA uses four bases: Adenine, Thymine, Guanine, and Cytosine.",         "category": "science",     "difficulty": "beginner"},
    {"text": "The Standard Model describes fundamental particles and forces.",         "category": "science",     "difficulty": "advanced"},
    # History
    {"text": "The French Revolution began in 1789 with the storming of the Bastille.","category": "history",    "difficulty": "beginner"},
    {"text": "The Industrial Revolution transformed manufacturing in 18th-century Britain.", "category": "history", "difficulty": "intermediate"},
    {"text": "World War II ended in 1945 with Allied victory in Europe and the Pacific.", "category": "history", "difficulty": "beginner"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Day 4 – Script 05: Full Semantic Search Engine")
    print("=" * 60)

    # ── Build engine ──────────────────────────────────────────────────────────
    engine = SemanticSearchEngine("all-MiniLM-L6-v2")
    engine.build_index(CORPUS)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n── Engine Statistics ──")
    stats = engine.stats()
    print(f"  Total documents : {stats['total_documents']}")
    print(f"  Embedding dim   : {stats['embedding_dim']}")
    print(f"  By category     : {stats['by_category']}")

    # ── Searches ──────────────────────────────────────────────────────────────
    def demo_search(query, k=3, filters=None):
        filter_str = f"  filters={filters}" if filters else ""
        print(f"\n❓ '{query}'{filter_str}")
        results = engine.search(query, k=k, filters=filters)
        if not results:
            print("  (no results)")
        for r in results:
            meta = f"[{r['category']} | {r['difficulty']}]"
            print(f"  [{r['score']:.4f}] {meta:30s} {r['text'][:65]}")

    print("\n──────────────────────────────────────────────────────────")
    print("Search demonstrations")
    print("──────────────────────────────────────────────────────────")

    demo_search("how does vector similarity search work?")
    demo_search("how does vector similarity search work?", filters={"category": "ai"})
    demo_search("building APIs and backend services", filters={"category": "programming"})
    demo_search("beginner-friendly science topics", filters={"difficulty": "beginner"})
    demo_search("advanced machine learning techniques", filters={"category": "ai", "difficulty": "advanced"})
    demo_search("major historical events in the 20th century", k=4)

    # ── Incremental add ───────────────────────────────────────────────────────
    print("\n── Incremental Document Addition ──")
    new_docs = [
        {"text": "Kubernetes orchestrates containerised applications at scale.", "category": "programming", "difficulty": "advanced"},
        {"text": "Climate change is accelerating due to greenhouse gas emissions.", "category": "science", "difficulty": "beginner"},
    ]
    engine.add_documents(new_docs)
    demo_search("container orchestration systems")

    # ── Save & Reload ─────────────────────────────────────────────────────────
    print("\n── Save & Reload Engine ──")
    engine.save("semantic_engine")

    engine2 = SemanticSearchEngine.__new__(SemanticSearchEngine)
    engine2.model    = engine.model   # reuse loaded model
    engine2.index    = None
    engine2.metadata = []
    engine2.d        = None
    engine2.load("semantic_engine")

    print("\nVerification search after reload:")
    for r in engine2.search("efficient vector indexing", k=2):
        print(f"  [{r['score']:.4f}] {r['text']}")

    # ── Clean up demo files ───────────────────────────────────────────────────
    for f in ["semantic_engine.faiss", "semantic_engine_meta.json",
              "corpus_index.faiss", "corpus_meta.json", "demo_index.faiss"]:
        if os.path.exists(f):
            os.remove(f)

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("""
──────────────────────────────────────────────────────────
Interactive Search REPL
  Type a query to search all documents.
  Prefix with @<category> to filter: @ai what are embeddings?
  Type 'exit' or 'quit' to stop.
──────────────────────────────────────────────────────────""")

    while True:
        try:
            raw = input("\nSearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw.lower() in ("exit", "quit", "q"):
            break

        # Parse optional category filter
        filters = None
        query   = raw
        if raw.startswith("@"):
            parts = raw.split(" ", 1)
            cat   = parts[0][1:]
            query = parts[1] if len(parts) > 1 else ""
            if query:
                filters = {"category": cat}
            else:
                print(f"  (please enter a query after @{cat})")
                continue

        if not query:
            continue

        results = engine.search(query, k=5, filters=filters)
        if not results:
            print("  No results found.")
        for r in results:
            meta = f"[{r['category']} | {r['difficulty']}]"
            print(f"  [{r['score']:.4f}] {meta:30s} {r['text']}")

    print("\n✅  Semantic search engine demo complete.")
