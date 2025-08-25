import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

KB_PATH = "Marine/DU-MobileYOLO-main/MarineExplainAI/rag/data/marine_kb_real.json"
BM25_INDEX_PATH = "Marine/DU-MobileYOLO-main/MarineExplainAI/rag/real_dataset/index/bm25_index_real.pkl"
ID_MAP_PATH = "Marine/DU-MobileYOLO-main/MarineExplainAI/rag/real_dataset/index/id_map_real.json"

with open(KB_PATH, 'r') as f:
    kb = json.load(f)

contexts = [entry["context"] for entry in kb]
ids = [entry["title"] for entry in kb]

vectorizer = TfidfVectorizer().fit(contexts)
bm25_index = vectorizer.transform(contexts)

with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump(bm25_index, f)

with open(ID_MAP_PATH, "w") as f:
    json.dump(ids, f)

print("✅ BM25 index and ID map created.")
