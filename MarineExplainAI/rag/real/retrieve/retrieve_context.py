import json
import pickle
import os
import re
from rank_bm25 import BM25Okapi

# === Absolute Paths (FIXED) ===
BASE_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI"
INDEX_PATH = os.path.join(BASE_DIR, "rag/index/bm25_index_real.pkl")
IDMAP_PATH = os.path.join(BASE_DIR, "rag/index/id_map_real.json")

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Load BM25 Index (Handle tuple case) ===
with open(INDEX_PATH, "rb") as f:
    loaded_obj = pickle.load(f)
    if isinstance(loaded_obj, tuple):
        bm25 = loaded_obj[0]
    else:
        bm25 = loaded_obj

with open(IDMAP_PATH, "r") as f:
    id_map = json.load(f)

# === Retrieval Function ===
def retrieve_context(query, top_k=3):
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": scores[idx],
            "title": id_map[str(idx)]["title"],
            "context": id_map[str(idx)]["context"]
        })
    return results

# === Test Run ===
if __name__ == "__main__":
    sample_query = "What do we know about the starfish found in shallow water?"
    top_contexts = retrieve_context(sample_query, top_k=3)

    for i, ctx in enumerate(top_contexts, 1):
        print(f"\n--- Context {i} (score={ctx['score']:.2f}) ---")
        print(f"Title: {ctx['title']}")
        print(f"Content:\n{ctx['context']}")
