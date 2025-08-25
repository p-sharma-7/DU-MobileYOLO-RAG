# File: rag/real/index/create_index_real.py

import json
import pickle
import os
from rank_bm25 import BM25Okapi
import re

# === Load Marine KB ===
KB_PATH = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/data/marine_kb_real.json"
with open(KB_PATH, "r") as f:
    kb = json.load(f)

# === Tokenizer ===
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Prepare Corpus ===
corpus = []
id_map = {}
for idx, entry in enumerate(kb):
    context = entry["context"]
    tokens = simple_tokenize(context)
    corpus.append(tokens)
    id_map[str(idx)] = {"title": entry["title"], "context": context}

# === BM25 Index ===
bm25 = BM25Okapi(corpus)

# === Save BM25 and ID Map ===
OUT_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/index"
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "bm25_index_real.pkl"), "wb") as f:
    pickle.dump(bm25, f)

with open(os.path.join(OUT_DIR, "id_map_real.json"), "w") as f:
    json.dump(id_map, f, indent=2)

print("✅ BM25 index and ID map saved successfully.")
