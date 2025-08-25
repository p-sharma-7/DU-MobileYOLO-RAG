import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === Input/Output Paths ===
INPUT_PROMPT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_quantum_real.csv"
OUTPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/explanations_quantum_real_sample30.csv"

# === Model Checkpoint ===
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

# === Load prompts ===
df = pd.read_csv(INPUT_PROMPT_CSV)
df_sample = df.sample(30, random_state=42).reset_index(drop=True)

results = []
for i, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    img_path = row["image_path"]
    prompt = row["prompt"]
    label = row["label"]

    try:
        image = Image.open(img_path).convert("RGB").resize((336, 336))
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({
            "image_path": img_path,
            "label": label,
            "prompt": prompt,
            "explanation": answer
        })

    except Exception as e:
        results.append({
            "image_path": img_path,
            "label": label,
            "prompt": prompt,
            "explanation": f"Error: {e}"
        })

    # Save every 5 samples
    if (i + 1) % 5 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

# Final save
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(results)} explanations to {OUTPUT_CSV}")
