import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# === CONFIGURATION ===
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
PROMPT_CSV = "results/method_2/prompts_m2_abs.csv"
OUTPUT_CSV = "results/method_2/explanations_m2_dialog.csv"
IMG_RESIZE = (336, 336)

# === Load model and processor ===
print("🚀 Loading LLaVA model...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
print("✅ Model loaded!")

# === Load prompt CSV ===
df = pd.read_csv(PROMPT_CSV)
print(f"📄 Loaded {len(df)} rows from {PROMPT_CSV}")

# === Load existing results if any ===
if os.path.exists(OUTPUT_CSV):
    existing_df = pd.read_csv(OUTPUT_CSV)
    completed_images = set(existing_df["original_image"])
    print(f"🔁 Resuming from {len(existing_df)} completed rows.")
else:
    existing_df = pd.DataFrame()
    completed_images = set()
    print("🆕 Starting fresh...")

# === Ensure output directory exists ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# === Start processing ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc="🧠 Generating Explanations"):
    img_path = row["original_image"]
    label = row["class_name"]

    if img_path in completed_images:
        continue

    if not os.path.exists(img_path):
        print(f"❌ Missing image: {img_path}")
        continue

    try:
        image = Image.open(img_path).convert("RGB").resize(IMG_RESIZE)
        prompt = f"<image>\nUSER: What marine species is likely visible in this scene containing a {label}?\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        explanation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Save result incrementally
        pd.DataFrame([{
            "original_image": img_path,
            "class_name": label,
            "prompt": prompt,
            "explanation": explanation
        }]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)

        completed_images.add(img_path)

    except Exception as e:
        print(f"❌ Generation failed at row {idx}: {e}")
        continue

print(f"\n✅ All explanations saved to: {OUTPUT_CSV}")
