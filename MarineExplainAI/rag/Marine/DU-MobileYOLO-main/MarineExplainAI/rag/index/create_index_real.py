import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📖 Reading knowledge base...")

KB_PATH = "data/marine_kb_real.json"  # ✅ FIXED relative path
INDEX_SAVE_PATH = "index/bm25_index_real.pkl"
ID_MAP_PATH = "index/id_map_real.json"

with open(KB_PATH, "r") as f:
    kb_entries = json.load(f)

documents = []
id_map = {}

for idx, entry in enumerate(kb_entries):
    species = entry.get("species_name", f"species_{idx}")
    traits = entry.get("traits", {})
    
    # Concatenate all available trait info into a single text doc
    doc = (
        f"Color: {traits.get('color', '')}. "
        f"Shape: {traits.get('body_shape', '')}. "
        f"Habitat: {traits.get('habitat', '')}. "
        f"Size: {traits.get('size', '')}. "
        f"Diet: {traits.get('diet', '')}. "
        f"Notable feature: {traits.get('special', '')}."
    )

    documents.append(doc)
    id_map[idx] = {
        "species_name": species,
        "text": doc,
    }

# Using Tfidf as an approximation for BM25
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Save index and ID map
with open(INDEX_SAVE_PATH, "wb") as f:
    pickle.dump((vectorizer, doc_vectors), f)

with open(ID_MAP_PATH, "w") as f:
    json.dump(id_map, f, indent=2)

print(f"✅ Indexed {len(documents)} entries using BM25!")
print(f"💾 Saved index to: {INDEX_SAVE_PATH}")
print(f"💾 Saved ID map to: {ID_MAP_PATH}")

