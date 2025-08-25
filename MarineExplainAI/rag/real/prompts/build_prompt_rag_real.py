import os
import pandas as pd
from tqdm import tqdm
import sys
import json

# === Fix relative import ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieve.retrieve_context import retrieve_context

# === Configuration ===
INPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/crop_metadata.csv"
OUTPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_rag_real.csv"
IMAGE_DIR = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images/"

# === Ensure output folder exists ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# === Load metadata ===
df = pd.read_csv(INPUT_CSV)
prompts = []

# === Generate prompts ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    species = row['class_name']
    image_file = row['original_image']
    abs_path = os.path.join(IMAGE_DIR, image_file)

    if not os.path.exists(abs_path):
        print(f"⚠️ File not found: {abs_path}")
        continue

    # === Retrieve context for the species ===
    try:
        contexts = retrieve_context(species, top_k=3)
        if not contexts:
            print(f"⚠️ No context found for {species}")
            continue
        context_text = "\n".join([f"- {ctx['context']}" for ctx in contexts])
    except Exception as e:
        print(f"⚠️ Retrieval failed for label '{species}': {e}")
        continue

    # === Construct the detailed prompt ===
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

    # Save prompt details
    prompts.append({
        "image_path": abs_path,
        "label": species,
        "prompt": prompt.strip()
    })

# === Save all prompts ===
out_df = pd.DataFrame(prompts)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(out_df)} prompts to {OUTPUT_CSV}")
