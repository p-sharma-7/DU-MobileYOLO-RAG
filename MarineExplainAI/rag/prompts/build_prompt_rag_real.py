import os
import pandas as pd
from rag.index.bm25_utils import load_bm25_index, load_id_map
from retrieve.retrieve_context import retrieve_context

# 📂 Paths
BM25_INDEX_PATH = "rag/index/bm25_index_real.pkl"
ID_MAP_PATH = "rag/index/id_map_real.json"
PROMPT_INPUT_CSV = "rag/5%_filtered_obis_species_processed_+common_species.csv"
OUTPUT_CSV_PATH = "results/rag_real/prompts_rag_real.csv"

# 📦 Load index
print("📦 Loading BM25 index and ID map...")
bm25 = load_bm25_index(BM25_INDEX_PATH)
id_map = load_id_map(ID_MAP_PATH)

# 🔄 Load input data
print("🔄 Generating prompts...")
df = pd.read_csv(PROMPT_INPUT_CSV)

prompts = []
for idx, row in df.iterrows():
    class_label = row["class_name_label"]
    image_id = row["image_id"] if "image_id" in row else idx
    retrieved_context = retrieve_context(query=class_label, bm25=bm25, id_map=id_map, top_k=3)
    
    prompt = f"<image>\nUSER: What marine species is likely visible in this scene containing a {class_label}?\nCONTEXT: {retrieved_context}\nASSISTANT:"
    prompts.append({"image_id": image_id, "class_label": class_label, "prompt": prompt})

# 💾 Save to CSV
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
pd.DataFrame(prompts).to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Saved {len(prompts)} prompts to: {OUTPUT_CSV_PATH}")
