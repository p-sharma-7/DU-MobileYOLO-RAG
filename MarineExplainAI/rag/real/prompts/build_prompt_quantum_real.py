import os
import sys
import pandas as pd
from rank_bm25 import BM25Okapi
import json

# === Append path so we can import quantum retriever ===
sys.path.append("/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/quantum/rag")

from quantum.quantum_retriever import retrieve_context
from quantum.quantum_encoder import simple_tokenize

# === Paths ===
CSV_PATH = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/real/data/abs_image_labels_real.csv"
ID_MAP_PATH = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/index/id_map.json"
OUTPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_quantum_real.csv"

# === Load Knowledge Base ID map ===
with open(ID_MAP_PATH, "r") as f:
    id_map = json.load(f)

# === Prepare BM25 Corpus ===
documents = [entry["context"] for entry in id_map.values()]
tokenized_docs = [simple_tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# === Load image-path and species CSV ===
df = pd.read_csv(CSV_PATH)
prompts = []

# === Construct prompts ===
for _, row in df.iterrows():
    image_path = row["image_path"]
    species = row["class_name"]

    # Search query
    query = f"{species} marine species biology and traits"

    # Retrieve top-k contextual info using quantum-enhanced RAG
    contexts = retrieve_context(query, bm25, id_map, top_k=3)
    context_text = "\n".join(contexts)

    # Final detailed prompt
    prompt = f"""<image>
USER: The object in the image is predicted to be a **{species}**.

Here are some biological facts and ecological context about this species:
{context_text}

Based on this information and the image, explain:
- Key visual features that support this identification
- Typical habitat and environment
- Ecological role, feeding behavior, and movement
- Distinctive traits for accurate classification

ASSISTANT:"""

    prompts.append({
        "image_path": image_path,
        "label": species,
        "prompt": prompt
    })

# === Save all prompts ===
output_df = pd.DataFrame(prompts)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(output_df)} quantum-enhanced prompts to {OUTPUT_CSV}")
