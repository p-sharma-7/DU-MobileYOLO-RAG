import pandas as pd
import os
import time
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

PROMPT_PATH = "results/rag_real/prompts_rag_real.csv"
OUTPUT_PATH = "results/rag_real/explanations_rag_real.csv"
MODEL_PATH = "vlm_models/llava_v1_5_7b/checkpoints"  # ✅ adjust if needed

# Load model
print("🔁 Loading LLaVA model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")

# Load prompts
df = pd.read_csv(PROMPT_PATH)

# Setup output
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results = []
start = 0

if os.path.exists(OUTPUT_PATH):
    existing = pd.read_csv(OUTPUT_PATH)
    processed_ids = set(existing['id'].tolist())
    df = df[~df['id'].isin(processed_ids)]
    results = existing.to_dict("records")
    start = len(results)

print(f"📊 Starting from index {start}")

# Generate explanations
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image_path = row["image_path"]
        prompt = row["prompt"]

        image = Image.open(image_path).convert("RGB").resize((336, 336))
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        output = model.generate(**inputs, max_new_tokens=128)
        response = processor.batch_decode(output, skip_special_tokens=True)[0]

        results.append({
            "id": row["id"],
            "image_path": image_path,
            "label": row["label"],
            "prompt": prompt,
            "explanation": response.strip()
        })

        # Save every 10 samples
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"💾 Saved {len(results)} samples...")

    except Exception as e:
        print(f"⚠️ Error processing id {row['id']}: {e}")

# Final save
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved final explanations to {OUTPUT_PATH}")
