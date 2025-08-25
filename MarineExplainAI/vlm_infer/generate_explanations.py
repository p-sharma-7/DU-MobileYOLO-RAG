# MarineExplainAI/vlm_infer/generate_explanations.py

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === Paths ===
PROMPT_CSV = "results/prompts_fixed.csv"
EXPLANATION_CSV = "results/explanations.csv"
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
FULL_IMAGE_DIR = "brackish_dataset/test/images"

# === Load Model & Processor ===
print("🚀 Loading LLaVA-1.5-7B model...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
print("✅ Model loaded!")

# === Load prompts ===
df = pd.read_csv(PROMPT_CSV)
outputs = []

# === Inference ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc="🧠 Generating Explanations"):
    crop_path = row["crop_path"]
    full_img_path = row["original_image"]
    prompt = row["prompt"]

    if not os.path.exists(crop_path) or not os.path.exists(full_img_path):
        print(f"❌ Missing image: {crop_path} or {full_img_path}")
        continue

    try:
        # Load both images
        crop_img = Image.open(crop_path).convert("RGB")
        full_img = Image.open(full_img_path).convert("RGB")

        # Create inputs for the model
        inputs = processor(
            images=[crop_img, full_img],
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        # Generate explanation
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Save result
        outputs.append({
            "crop_path": crop_path,
            "original_image": row["original_image"],
            "class_name": row["class_name"],
            "prompt": prompt,
            "explanation": output_text
        })

    except Exception as e:
        print(f"❌ Error on row {idx}: {e}")
        continue

# === Save CSV ===
output_df = pd.DataFrame(outputs)
output_df.to_csv(EXPLANATION_CSV, index=False)
print(f"\n✅ All explanations saved to: {EXPLANATION_CSV}")
