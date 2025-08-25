import json
import numpy as np

from .quantum_encoder import text_to_quantum_angles, simple_tokenize
from .quantum_similarity import compute_similarity

# === Load KB ID map ===
with open("/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/index/id_map.json", "r") as f:
    id_map = json.load(f)

# === Precompute document encodings ===
kb_states = {}
for idx, entry in id_map.items():
    context = entry["context"]
    angles = text_to_quantum_angles(context)
    kb_states[idx] = angles

# === Quantum-Aware Retriever ===
def retrieve_context(query, bm25, id_map, top_k=3):
    # Step 1: Tokenize and get BM25 scores
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    bm25_top_k_ids = np.argsort(scores)[::-1][:top_k * 2]  # Take more for reranking

    # Step 2: Encode query to quantum state
    query_angles = text_to_quantum_angles(query)

    # Step 3: Rerank top BM25 results by quantum similarity
    scored = []
    for idx in bm25_top_k_ids:
        idx_str = str(idx)
        if idx_str in kb_states:
            doc_angles = kb_states[idx_str]
            sim_score = compute_similarity(query_angles, doc_angles)
            scored.append((idx_str, sim_score))

    # Step 4: Sort and return top-k most similar entries
    top_indices = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [id_map[idx]["context"] for idx, _ in top_indices]
