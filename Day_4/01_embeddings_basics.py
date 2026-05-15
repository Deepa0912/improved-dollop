"""
Day 4 – Script 01: Embeddings Basics with Sentence Transformers
================================================================
This script demonstrates:
  - Loading a Sentence Transformer model
  - Generating sentence embeddings
  - Inspecting embedding shapes and values
  - Comparing embeddings visually (dot-product heatmap without matplotlib)
  - Showing how similar sentences cluster together

Run:
    python 01_embeddings_basics.py

Requirements:
    pip install sentence-transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Day 4 – Embeddings Basics with Sentence Transformers")
print("=" * 60)

MODEL_NAME = "all-MiniLM-L6-v2"   # 90MB, 384-dim, fast & free
print(f"\nLoading model: {MODEL_NAME} (downloads on first run)...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Encode a few sentences
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. Single Sentence Embedding ──")

sentence = "Artificial intelligence is transforming the world."
embedding = model.encode(sentence)

print(f"Sentence    : {sentence}")
print(f"Shape       : {embedding.shape}")          # (384,)
print(f"Dtype       : {embedding.dtype}")
print(f"Min / Max   : {embedding.min():.4f} / {embedding.max():.4f}")
print(f"Norm (L2)   : {np.linalg.norm(embedding):.4f}")   # ~1 if normalised
print(f"First 8 dims: {embedding[:8].round(4)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Batch encoding
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. Batch Encoding ──")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",       # semantically similar to #0
    "Dogs love to play fetch.",
    "The stock market crashed today.",
    "Machine learning uses data to make predictions.",
    "Neural networks are inspired by the human brain.",
]

embeddings = model.encode(sentences, show_progress_bar=False)
print(f"Input sentences : {len(sentences)}")
print(f"Embeddings shape: {embeddings.shape}")   # (6, 384)

# ─────────────────────────────────────────────────────────────────────────────
# 4. L2-normalised embeddings (unit vectors)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. L2-Normalised Embeddings ──")

norm_embs = model.encode(sentences, normalize_embeddings=True)
norms = np.linalg.norm(norm_embs, axis=1)
print(f"All norms after normalisation: {norms.round(4)}")
print(f"(Should all be ~1.0000)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Raw dot-product similarity matrix (since vectors are L2-normalised,
#    dot product == cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Pairwise Similarity (dot-product on normalised vectors) ──")

sim_matrix = norm_embs @ norm_embs.T   # (6, 6)
print("\nSimilarity matrix (rows/cols = sentences 0-5):\n")

# Print nicely aligned
header = "        " + "  ".join(f"  s{i}" for i in range(len(sentences)))
print(header)
for i, row in enumerate(sim_matrix):
    row_str = "  ".join(f"{v:5.3f}" for v in row)
    print(f"  s{i}  [ {row_str} ]")

print("\nKey observations:")
print(f"  s0 ↔ s1 (cat/feline)   : {sim_matrix[0,1]:.4f}  ← should be HIGH")
print(f"  s4 ↔ s5 (ML/neural)    : {sim_matrix[4,5]:.4f}  ← should be HIGH")
print(f"  s0 ↔ s3 (cat/stock mkt): {sim_matrix[0,3]:.4f}  ← should be LOW")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparing different model sizes
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Model Comparison ──")

model_info = [
    ("all-MiniLM-L6-v2",     "384-dim, fast, 90MB"),
    ("all-mpnet-base-v2",    "768-dim, high quality, 420MB"),
]
print(f"{'Model':<35} {'Dimension':<12} {'Note'}")
print("-" * 70)
for name, note in model_info:
    try:
        m = SentenceTransformer(name)
        dim = m.encode("test").shape[0]
        print(f"{name:<35} {dim:<12} {note}")
    except Exception as e:
        print(f"{name:<35} {'N/A':<12} (load error: {e})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Encode a large batch efficiently
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. Large Batch Encoding ──")

# Simulate a larger corpus
large_corpus = [f"This is sentence number {i} about topic {i % 5}." for i in range(100)]
model_fast = SentenceTransformer("all-MiniLM-L6-v2")

import time
start = time.time()
large_embs = model_fast.encode(
    large_corpus,
    batch_size=32,
    show_progress_bar=False,
    normalize_embeddings=True,
)
elapsed = time.time() - start

print(f"Encoded {len(large_corpus)} sentences in {elapsed:.2f}s")
print(f"Throughput: {len(large_corpus)/elapsed:.0f} sentences/sec")
print(f"Output shape: {large_embs.shape}")

print("\n✅  Embeddings basics demo complete.")
