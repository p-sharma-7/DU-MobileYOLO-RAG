import json
import pickle
import os
from rank_bm25 import BM25Okapi
import re

# === CONFIGURATION ===
KB_JSON = "rag/data/marine_kb_real.json"
INDEX_OUT = "rag/index/bm25_index_real.pkl"
IDMAP_OUT = "rag/index/id_map_real.json"

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def load_kb(path):
    with open(path, "r") as f:
        data = json.load(f)

    docs = []
    id_map = {}
    for idx, entry in enumerate(data):
        content = entry["context"]
        docs.append(content)
        id_map[idx] = {
            "title": entry.get("title", f"doc_{idx}"),
            "context": content
        }
    return docs, id_map

def build_bm25_index(docs):
    tokenized = [simple_tokenize(doc) for doc in docs]
    return BM25Okapi(tokenized)

def save_index(bm25, id_map):
    with open(INDEX_OUT, "wb") as f:
        pickle.dump(bm25, f)
    with open(IDMAP_OUT, "w") as f:
        json.dump(id_map, f, indent=2)

def main():
    print("📦 Loading knowledge base...")
    docs, id_map = load_kb(KB_JSON)

    print("⚙️ Building BM25 index...")
    bm25 = build_bm25_index(docs)

    print("💾 Saving index...")
    save_index(bm25, id_map)

    print(f"✅ BM25 index saved to: {INDEX_OUT}")
    print(f"✅ ID Map saved to: {IDMAP_OUT}")

if __name__ == "__main__":
    main()
