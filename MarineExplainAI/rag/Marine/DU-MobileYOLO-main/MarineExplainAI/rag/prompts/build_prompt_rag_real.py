import os
import json
import pandas as pd
from tqdm import tqdm
from rag.retrieve.retrieve_context import retrieve_context
from rag.index.bm25_utils import load_bm25_index

# 🔧 Paths
BM25_INDEX_PATH = "rag/index/bm25_index_real.pkl"
ID_MAP_PATH = "rag/index/id_map_real.json"
PROMPT_INPUT_CSV = "Marine/DU-MobileYOLO-main/MarineExplainAI/results/method_2/prompts_m2_abs.csv"
PROMPT_OUTPUT_CSV = "Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_rag_real.csv"

# 🔃 Load BM25
print("📦 Loading BM25 index and ID map...")
bm25, id_map = load_bm25_index(BM25_INDEX_PATH, ID_MAP_PATH)

# 🔃 Load image data
df = pd.read_csv(PROMPT_INPUT_CSV)
df = df[['original_image', 'class_name']]  # Use only necessary columns
df = df.dropna().drop_duplicates()

# 📄 Prepare output CSV
os.makedirs(os.path.dirname(PROMPT_OUTPUT_CSV), exist_ok=True)

print("🔄 Generating prompts...")
prompt_records = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['original_image']
    label = row['class_name'].strip().lower()

    # Get top 3 facts
    retrieved_facts = retrieve_context(label, bm25_index=bm25, id_map=id_map, top_k=3)
    context = "Knowledge Context:\n" + "\n".join(retrieved_facts) if retrieved_facts else "Knowledge Context:\nNo context found."

    # Final prompt template
    prompt = f"""{context}

USER: Based on the image and the knowledge above, explain why this image likely contains a {label}.
ASSISTANT:"""

    prompt_records.append({
        "image_path": img_path,
        "class_label": label,
        "prompt": prompt
    })

    # Save every 10 prompts
    if len(prompt_records) % 10 == 0:
        pd.DataFrame(prompt_records).to_csv(PROMPT_OUTPUT_CSV, index=False)

# Final save
pd.DataFrame(prompt_records).to_csv(PROMPT_OUTPUT_CSV, index=False)
print(f"✅ Saved {len(prompt_records)} prompts to: {PROMPT_OUTPUT_CSV}")

