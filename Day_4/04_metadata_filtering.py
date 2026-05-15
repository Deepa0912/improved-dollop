"""
Day 4 – Script 04: Metadata Filtering with FAISS
=================================================
FAISS stores only vectors — no metadata.
This script demonstrates the standard pattern for adding metadata:

  1. Maintain a parallel Python list (metadata store)
  2. Post-filter results by metadata field
  3. Pre-filter pattern (restrict candidate set before FAISS search)
  4. Combining multiple filter conditions (AND / OR logic)
  5. Persisting metadata alongside the FAISS index (JSON)

Run:
    python 04_metadata_filtering.py

Requirements:
    pip install sentence-transformers faiss-cpu
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Day 4 – Script 04: Metadata Filtering with FAISS")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# Our "database" — a list of dicts with text + metadata
# ─────────────────────────────────────────────────────────────────────────────
corpus = [
    # --- AI / ML ---
    {"text": "FAISS allows fast nearest-neighbour search at scale.",         "category": "ai",       "level": "advanced", "year": 2023, "lang": "en"},
    {"text": "Sentence Transformers create dense sentence embeddings.",       "category": "ai",       "level": "beginner", "year": 2022, "lang": "en"},
    {"text": "PyTorch is widely used for training neural networks.",          "category": "ai",       "level": "intermediate", "year": 2022, "lang": "en"},
    {"text": "Cosine similarity finds semantically related documents.",       "category": "ai",       "level": "beginner", "year": 2023, "lang": "en"},
    {"text": "Large language models are trained on massive text corpora.",     "category": "ai",       "level": "advanced", "year": 2023, "lang": "en"},
    {"text": "ChromaDB is an open-source vector database for embeddings.",    "category": "ai",       "level": "intermediate", "year": 2023, "lang": "en"},
    # --- Travel ---
    {"text": "The Eiffel Tower is a symbol of Paris and French culture.",     "category": "travel",   "level": "beginner", "year": 2021, "lang": "en"},
    {"text": "Kyoto is famous for its temples and traditional geisha culture.","category": "travel",  "level": "beginner", "year": 2021, "lang": "en"},
    {"text": "Santorini is a volcanic island known for its white buildings.",  "category": "travel",   "level": "beginner", "year": 2022, "lang": "en"},
    {"text": "The Amazon rainforest spans nine South American countries.",     "category": "travel",   "level": "intermediate", "year": 2022, "lang": "en"},
    # --- Science ---
    {"text": "Quantum entanglement allows particles to correlate over distances.", "category": "science", "level": "advanced", "year": 2023, "lang": "en"},
    {"text": "DNA replication ensures genetic information is passed to daughter cells.", "category": "science", "level": "intermediate", "year": 2022, "lang": "en"},
    {"text": "Black holes form when massive stars collapse under their gravity.", "category": "science", "level": "beginner", "year": 2021, "lang": "en"},
    # --- Programming ---
    {"text": "Python decorators are a powerful metaprogramming tool.",        "category": "programming", "level": "intermediate", "year": 2022, "lang": "en"},
    {"text": "JavaScript async/await simplifies asynchronous code.",          "category": "programming", "level": "beginner",      "year": 2023, "lang": "en"},
    {"text": "Rust guarantees memory safety without garbage collection.",      "category": "programming", "level": "advanced",      "year": 2023, "lang": "en"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Build the FAISS index (pure vector storage)
# ─────────────────────────────────────────────────────────────────────────────
texts      = [doc["text"] for doc in corpus]
embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
d          = embeddings.shape[1]

index = faiss.IndexFlatIP(d)
index.add(embeddings)
print(f"\nIndex ready: {index.ntotal} vectors of dim {d}")
print(f"Metadata store: {len(corpus)} entries")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Post-filter search function
# ─────────────────────────────────────────────────────────────────────────────
def search(
    query: str,
    k: int = 5,
    filters: dict = None,    # {"category": "ai"} or {"level": "beginner", "year": 2023}
    logic: str = "AND",      # "AND" or "OR"
) -> list[dict]:
    """
    Embed query, retrieve top candidates from FAISS,
    then post-filter by metadata.
    """
    query_emb  = model.encode([query], normalize_embeddings=True).astype("float32")
    candidates = min(index.ntotal, k * 6)   # over-fetch to allow for filtering
    distances, indices = index.search(query_emb, candidates)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        doc = corpus[idx]
        if filters:
            matches = [doc.get(fk) == fv for fk, fv in filters.items()]
            if logic == "AND" and not all(matches):
                continue
            if logic == "OR" and not any(matches):
                continue
        results.append({"score": round(float(dist), 4), **doc})
        if len(results) >= k:
            break
    return results

def print_results(title: str, results: list):
    print(f"\n── {title} ──")
    if not results:
        print("  (no results after filtering)")
        return
    for r in results:
        meta = f"[{r['category']} | {r['level']} | {r['year']}]"
        print(f"  [{r['score']:.4f}] {meta:35s} {r['text'][:60]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Demo: no filter vs filtered
# ─────────────────────────────────────────────────────────────────────────────
QUERY = "fast search over high-dimensional vectors"

print_results(
    "No filter",
    search(QUERY, k=4),
)

print_results(
    "Filter: category=ai",
    search(QUERY, k=4, filters={"category": "ai"}),
)

print_results(
    "Filter: category=ai AND level=beginner",
    search(QUERY, k=4, filters={"category": "ai", "level": "beginner"}),
)

print_results(
    "Filter: category=ai AND year=2023",
    search(QUERY, k=4, filters={"category": "ai", "year": 2023}),
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. OR logic
# ─────────────────────────────────────────────────────────────────────────────
print_results(
    "Filter: category=travel OR category=science  (OR logic)",
    search("interesting natural phenomena", k=4,
           filters={"category": "travel", "level": "advanced"}, logic="OR"),
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-filter pattern (build a sub-index for a category)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Pre-filter Pattern: Build Sub-Index per Category ──")

def build_subindex(category: str):
    """Return a FAISS index and local metadata list for one category."""
    subset     = [(i, doc) for i, doc in enumerate(corpus) if doc["category"] == category]
    local_meta = [doc for _, doc in subset]
    local_embs = np.stack([embeddings[i] for i, _ in subset]).astype("float32")
    sub_idx    = faiss.IndexFlatIP(d)
    sub_idx.add(local_embs)
    return sub_idx, local_meta

ai_index, ai_meta = build_subindex("ai")
print(f"AI sub-index contains {ai_index.ntotal} vectors")

q_emb   = model.encode([QUERY], normalize_embeddings=True).astype("float32")
dists, idxs = ai_index.search(q_emb, 3)

print(f"\nTop-3 AI results for: '{QUERY}'")
for dist, idx in zip(dists[0], idxs[0]):
    print(f"  [{dist:.4f}] {ai_meta[idx]['text']}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Persist index + metadata to disk
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Persisting Index + Metadata ──")

INDEX_PATH = "corpus_index.faiss"
META_PATH  = "corpus_meta.json"

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

print(f"Saved FAISS index → {INDEX_PATH}")
print(f"Saved metadata    → {META_PATH}")

# Reload
loaded_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, encoding="utf-8") as f:
    loaded_meta = json.load(f)

print(f"Reloaded: {loaded_index.ntotal} vectors, {len(loaded_meta)} metadata entries")

# Verify
dists2, idxs2 = loaded_index.search(q_emb, 3)
print("\nVerification search top-1 after reload:")
print(f"  [{dists2[0][0]:.4f}] {loaded_meta[idxs2[0][0]]['text']}")

print("""
Key Takeaways – Metadata Filtering with FAISS:
  ─ FAISS stores only float32 vectors (no text, no metadata)
  ─ Always maintain a parallel Python list of metadata dicts
  ─ Document position in corpus list == FAISS internal ID
  ─ Post-filter: over-fetch (k*4 or k*6), then apply filters
  ─ Pre-filter: build a separate index per category (faster for big corpora)
  ─ For built-in metadata filtering → use ChromaDB or Weaviate instead
""")

print("✅  Metadata filtering demo complete.")
