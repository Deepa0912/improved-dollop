"""
Day 4 – Script 03: FAISS Similarity Search
===========================================
This script demonstrates:
  1. IndexFlatL2  – exact L2 (Euclidean) distance search
  2. IndexFlatIP  – exact Inner Product (cosine sim after L2-norm)
  3. IndexIVFFlat – approximate search (faster for large corpora)
  4. IndexHNSWFlat – graph-based ANN search
  5. Save and reload an index to / from disk
  6. Benchmarking: exact vs approximate at scale

Run:
    python 03_faiss_search.py

Requirements:
    pip install sentence-transformers faiss-cpu
"""

import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Day 4 – Script 03: FAISS Similarity Search")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus and embeddings
# ─────────────────────────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    # Tech / AI
    "FAISS enables fast similarity search over millions of vectors.",
    "Sentence Transformers produce dense semantic embeddings.",
    "PyTorch is a popular deep learning framework developed by Meta.",
    "TensorFlow is Google's open-source machine learning platform.",
    "Cosine similarity measures the angle between two embedding vectors.",
    "Transformers revolutionised NLP with the attention mechanism.",
    "BERT is a powerful bidirectional encoder from Google.",
    "GPT models generate text by predicting the next token.",
    "Vector databases store and retrieve high-dimensional embeddings.",
    "RAG combines retrieval with large language model generation.",
    # Travel
    "Paris is the capital of France and home to the Eiffel Tower.",
    "Rome has ancient ruins including the Colosseum and the Forum.",
    "Tokyo is a bustling city blending tradition and modernity.",
    "New York City is famous for Central Park and Times Square.",
    "Barcelona is known for Gaudí's unique architecture.",
    # Science
    "Quantum computing uses qubits to process information.",
    "Black holes are regions where gravity is so strong light cannot escape.",
    "DNA carries genetic information using four nucleotide bases.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The speed of light in a vacuum is approximately 3×10⁸ m/s.",
]

print(f"\nEncoding {len(documents)} documents...")
embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
embeddings = embeddings.astype("float32")
d          = embeddings.shape[1]
print(f"Embedding shape: {embeddings.shape}  (n_docs × dim)")

QUERY = "How do vector search engines work?"
q_emb = model.encode([QUERY], normalize_embeddings=True).astype("float32")

K = 5  # top-k results

def print_results(label: str, distances, indices, docs):
    print(f"\n── {label} ──")
    print(f"Query: '{QUERY}'\n")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx == -1:
            continue
        print(f"  {rank}. [{dist:.4f}] {docs[idx]}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. IndexFlatL2 – Exact L2 distance (smaller = more similar)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. IndexFlatL2 (Exact Euclidean Distance) ──")

index_l2 = faiss.IndexFlatL2(d)
index_l2.add(embeddings)

distances_l2, indices_l2 = index_l2.search(q_emb, K)
print_results("IndexFlatL2 results (lower distance = more similar)", distances_l2, indices_l2, documents)

# ─────────────────────────────────────────────────────────────────────────────
# 2. IndexFlatIP – Exact Inner Product (cosine sim for L2-normalised vectors)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. IndexFlatIP (Exact Cosine Similarity) ──")

index_ip = faiss.IndexFlatIP(d)
index_ip.add(embeddings)

distances_ip, indices_ip = index_ip.search(q_emb, K)
print_results("IndexFlatIP results (higher score = more similar)", distances_ip, indices_ip, documents)

# ─────────────────────────────────────────────────────────────────────────────
# 3. IndexIVFFlat – Approximate, faster for large corpora
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. IndexIVFFlat (Approximate, Inverted File Index) ──")

nlist     = 5       # Number of Voronoi cells (clusters)
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# IVF index must be trained before adding vectors
print("Training IVF index...")
index_ivf.train(embeddings)
index_ivf.add(embeddings)

index_ivf.nprobe = 3   # Number of clusters to search (higher = more accurate, slower)

distances_ivf, indices_ivf = index_ivf.search(q_emb, K)
print_results(f"IndexIVFFlat (nlist={nlist}, nprobe={index_ivf.nprobe})", distances_ivf, indices_ivf, documents)

print(f"\n  IVF nprobe={index_ivf.nprobe}: searches {index_ivf.nprobe}/{nlist} clusters")
print("  → Increase nprobe for higher accuracy, decrease for speed")

# ─────────────────────────────────────────────────────────────────────────────
# 4. IndexHNSWFlat – Graph-based ANN (very fast, high accuracy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. IndexHNSWFlat (Hierarchical Navigable Small World) ──")

M         = 16    # Number of connections per node in the graph (higher = more accurate)
index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.add(embeddings)

distances_hnsw, indices_hnsw = index_hnsw.search(q_emb, K)
print_results(f"IndexHNSWFlat (M={M})", distances_hnsw, indices_hnsw, documents)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save and reload a FAISS index
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Save & Reload FAISS Index ──")

faiss.write_index(index_ip, "demo_index.faiss")
print("Index saved to demo_index.faiss")

loaded_index = faiss.read_index("demo_index.faiss")
print(f"Index reloaded: {loaded_index.ntotal} vectors, dim={loaded_index.d}")

# Verify search still works
d_r, i_r = loaded_index.search(q_emb, 3)
print(f"Verification search top-1: [{d_r[0][0]:.4f}] {documents[i_r[0][0]]}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Benchmark: Exact (FlatIP) vs Approximate (IVF) at scale
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. Benchmark: Exact vs Approximate Search ──")

# Simulate a larger corpus (1000 synthetic vectors)
rng        = np.random.default_rng(42)
big_vecs   = rng.standard_normal((1_000, d)).astype("float32")
# L2-normalise
norms_big  = np.linalg.norm(big_vecs, axis=1, keepdims=True)
big_vecs  /= norms_big

big_query  = big_vecs[:1]   # Use first vector as query
K_bench    = 10

# Exact
idx_exact = faiss.IndexFlatIP(d)
idx_exact.add(big_vecs)
t0 = time.perf_counter()
for _ in range(20):
    idx_exact.search(big_query, K_bench)
exact_ms = (time.perf_counter() - t0) / 20 * 1000

# IVF Approximate
nlist_bench = 50
q_bench = faiss.IndexFlatIP(d)
idx_approx = faiss.IndexIVFFlat(q_bench, d, nlist_bench, faiss.METRIC_INNER_PRODUCT)
idx_approx.train(big_vecs)
idx_approx.add(big_vecs)
idx_approx.nprobe = 10

t0 = time.perf_counter()
for _ in range(20):
    idx_approx.search(big_query, K_bench)
approx_ms = (time.perf_counter() - t0) / 20 * 1000

print(f"\n  Dataset size  : 1,000 vectors × {d} dims")
print(f"  IndexFlatIP   : {exact_ms:.3f} ms/query  (exact)")
print(f"  IndexIVFFlat  : {approx_ms:.3f} ms/query  (approx, nprobe={idx_approx.nprobe})")
print(f"  Speedup       : {exact_ms/approx_ms:.1f}x  (grows with dataset size)")

print("""
FAISS Index Cheat Sheet:
  IndexFlatL2   → Exact, L2 distance     → Best accuracy, slowest
  IndexFlatIP   → Exact, cosine sim      → Best accuracy for normalised vectors
  IndexIVFFlat  → Approx (IVF)           → Good balance accuracy/speed for 10k+ docs
  IndexHNSWFlat → Approx (graph-based)   → Very fast queries, higher memory
""")

print("✅  FAISS similarity search demo complete.")
