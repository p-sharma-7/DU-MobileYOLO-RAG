import os
import pandas as pd
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === Config ===
INPUT_PROMPT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_rag_real.csv"
OUTPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/explanations_rag_real_sample30.csv"
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
SAVE_EVERY = 5
NUM_SAMPLES = 30

# === Load model & processor ===
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

# === Load prompts ===
df = pd.read_csv(INPUT_PROMPT_CSV)
df_sampled = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)

# === Generate responses ===
results = []
for i, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
    img_path = row["image_path"]
    prompt = row["prompt"]
    label = row["label"]

    if not os.path.exists(img_path):
        explanation = f"Error: Image not found at {img_path}"
    else:
        try:
            image = Image.open(img_path).convert("RGB").resize((336, 336))
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            explanation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except UnidentifiedImageError:
            explanation = f"Error: Cannot open image {img_path}"
        except Exception as e:
            explanation = f"Error: {e}"

    results.append({
        "image_path": img_path,
        "label": label,
        "prompt": prompt,
        "explanation": explanation
    })

    # === Incremental Save ===
    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == NUM_SAMPLES:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Saved {i+1} / {NUM_SAMPLES} entries")

print(f"\n🎯 Final CSV saved to:\n{OUTPUT_CSV}")
