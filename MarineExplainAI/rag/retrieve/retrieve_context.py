import json
import pickle
import os
from rank_bm25 import BM25Okapi
import re

# === CONFIGURATION ===
INDEX_PATH = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/index/bm25_index_real.pkl"
IDMAP_PATH = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/index/id_map_real.json"

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Load BM25 Index and ID Map ===
with open(INDEX_PATH, "rb") as f:
    bm25 = pickle.load(f)

with open(IDMAP_PATH, "r") as f:
    id_map = json.load(f)

# === Retrieval Function ===
def retrieve_context(query, top_k=3):
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        item = id_map[str(idx)]
        results.append({
            "score": scores[idx],
            "title": item["title"],
            "context": item["context"]
        })
    return results

# === Optional: Test Run ===
if __name__ == "__main__":
    sample_query = "crab"
    top_contexts = retrieve_context(sample_query, top_k=3)

    for i, ctx in enumerate(top_contexts, 1):
        print(f"\n--- Context {i} (score={ctx['score']:.2f}) ---")
        print(f"Title: {ctx['title']}")
        print(f"Content:\n{ctx['context']}")
