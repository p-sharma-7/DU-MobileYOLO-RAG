import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq

# === Config ===
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
IMG_RESIZE = (336, 336)

# === Real test sample ===
img_path = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images/2019-03-06_22-12-29to2019-03-06_22-12-37_1-0020_jpg.rf.e87cf9dfc173132d6275ea6902b1125d.jpg"
label = "crab"
joined_context = (
    "Sea urchins are spiny marine animals that live on the ocean floor and graze on algae.\n"
    "---\n"
    "Crabs are crustaceans with a hard exoskeleton and pincers, commonly found in coastal regions and tidal pools.\n"
    "---\n"
    "Sea cucumbers are soft-bodied echinoderms that live on the seafloor and recycle nutrients."
)

prompt = (
    f"You are a marine biology visual expert.\n"
    f"Knowledge Context:\n{joined_context}\n"
    f"<image>\n"
    f"USER: Based on the image and the knowledge above, explain why this image likely contains a {label}.\n"
    f"ASSISTANT:"
)

# === Load image ===
print(f"🖼️ Loading image: {img_path}")
image = Image.open(img_path).convert("RGB").resize(IMG_RESIZE)

# === Load model + processor ===
print("🚀 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
model.eval()
print("✅ Model ready!")

# === Inference ===
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cpu")

print("💬 Generating explanation...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# === Output ===
print("\n📢 Explanation Output:\n")
print(f"USER: {prompt.split('USER: ')[-1]}")
print(f"ASSISTANT: {output}")
