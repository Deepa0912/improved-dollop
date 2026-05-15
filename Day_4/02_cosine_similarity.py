"""
Day 4 – Script 02: Cosine Similarity Deep Dive
===============================================
This script demonstrates:
  1. Manual cosine similarity computation (numpy)
  2. Pairwise similarity matrix with sklearn
  3. sentence_transformers util.cos_sim (PyTorch tensors)
  4. Finding top-k most similar sentences to a query
  5. Semantic Textual Similarity (STS) scoring
  6. The effect of L2 normalisation on dot product

Run:
    python 02_cosine_similarity.py

Requirements:
    pip install sentence-transformers scikit-learn
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("Day 4 – Script 02: Cosine Similarity")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Manual cosine similarity using numpy
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. Manual Cosine Similarity (numpy) ──")

def cosine_sim_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    dot   = np.dot(a, b)
    norm  = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0

pairs = [
    ("I love playing football.",   "Soccer is my favourite sport."),    # similar
    ("The sun is a star.",         "Stars are massive balls of gas."),  # related
    ("I love playing football.",   "The recipe requires two eggs."),    # unrelated
]

for s1, s2 in pairs:
    e1, e2 = model.encode([s1, s2])
    score  = cosine_sim_numpy(e1, e2)
    print(f"  {score:+.4f}  |  '{s1[:35]}…' ↔ '{s2[:35]}…'")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pairwise similarity matrix (sklearn)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. Pairwise Similarity Matrix (sklearn) ──")

sentences = [
    "Deep learning is a type of machine learning.",
    "Machine learning enables computers to learn from data.",
    "The Eiffel Tower stands 330 metres tall.",
    "Neural networks are the foundation of deep learning.",
    "Paris is known for its iconic tower.",
]

embs   = model.encode(sentences)
matrix = cosine_similarity(embs)   # shape (5, 5)

labels = [f"s{i}" for i in range(len(sentences))]
print(f"\n{'':6s}" + "  ".join(f"{l:>6}" for l in labels))
for i, row in enumerate(matrix):
    row_str = "  ".join(f"{v:6.3f}" for v in row)
    print(f"  {labels[i]}  {row_str}")

print("\nSentences (for reference):")
for i, s in enumerate(sentences):
    print(f"  s{i}: {s}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. sentence_transformers util.cos_sim (PyTorch tensor API)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. util.cos_sim (sentence_transformers PyTorch API) ──")

query      = "What is deep learning?"
candidates = [
    "Deep learning is a subset of machine learning using neural networks.",
    "The Great Wall of China stretches thousands of miles.",
    "Neural networks learn hierarchical representations.",
    "I enjoy hiking in the mountains.",
    "Backpropagation is the algorithm used to train neural networks.",
]

query_emb     = model.encode(query,      convert_to_tensor=True)
candidate_embs = model.encode(candidates, convert_to_tensor=True)

scores = util.cos_sim(query_emb, candidate_embs)   # (1, 5) tensor

print(f"\nQuery: '{query}'\n")
print(f"{'Rank':<5} {'Score':>7}  Candidate")
print("-" * 70)

# Sort by score descending
ranked = sorted(enumerate(scores[0].tolist()), key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(ranked, 1):
    print(f"  {rank:<3}  {score:+.4f}  {candidates[idx]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-k most similar pairs in a corpus (semantic duplicate detection)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Finding Semantic Duplicates / Near-Duplicates ──")

corpus = [
    "How do I install Python?",
    "What is the best way to set up Python?",
    "The moon orbits the Earth.",
    "Earth is orbited by the Moon.",
    "How do neural networks work?",
    "Can you explain how neural nets function?",
    "What is the capital of France?",
    "Where is Paris located?",
]

corpus_embs = model.encode(corpus, normalize_embeddings=True)
sim         = cosine_similarity(corpus_embs)

THRESHOLD = 0.75
print(f"\nPairs with cosine similarity ≥ {THRESHOLD}:\n")
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        score = sim[i, j]
        if score >= THRESHOLD:
            print(f"  [{score:.4f}]")
            print(f"    A: {corpus[i]}")
            print(f"    B: {corpus[j]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. L2 normalisation: dot product == cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Dot Product vs Cosine Sim (after L2 normalisation) ──")

s1, s2 = "I love cats.", "Cats are my favourite animals."
e1_raw = model.encode([s1, s2])
e1_norm = model.encode([s1, s2], normalize_embeddings=True)

raw_dot    = float(e1_raw[0] @ e1_raw[1])
norm_dot   = float(e1_norm[0] @ e1_norm[1])
cos_manual = cosine_sim_numpy(e1_raw[0], e1_raw[1])

print(f"  Raw dot product               : {raw_dot:.6f}")
print(f"  Cosine similarity (manual)    : {cos_manual:.6f}")
print(f"  Dot product after L2-norm     : {norm_dot:.6f}")
print(f"  All three match (for unit vec): {abs(cos_manual - norm_dot) < 1e-5}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Similarity score interpretation guide
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. Score Interpretation Guide ──")

test_pairs = [
    ("I love cats.", "I love cats."),                                  # identical
    ("I love cats.", "I adore felines."),                              # paraphrase
    ("Python is a programming language.", "Python can be venomous."), # polysemy
    ("I love cats.", "The economy grew by 3% last quarter."),          # unrelated
]

print(f"\n{'Score':>7}  {'Label':<15}  Pair")
print("-" * 75)
for s1, s2 in test_pairs:
    e1, e2 = model.encode([s1, s2], normalize_embeddings=True)
    score  = float(e1 @ e2)
    if score >= 0.9:
        label = "near-identical"
    elif score >= 0.7:
        label = "very similar"
    elif score >= 0.5:
        label = "somewhat similar"
    elif score >= 0.3:
        label = "weakly related"
    else:
        label = "unrelated"
    print(f"  {score:+.4f}  {label:<15}  '{s1[:25]}' ↔ '{s2[:25]}'")

print("\n✅  Cosine similarity demo complete.")
