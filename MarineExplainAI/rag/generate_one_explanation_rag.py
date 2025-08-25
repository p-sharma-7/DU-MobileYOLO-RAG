import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === CONFIG ===
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
IMG_PATH = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images/2019-02-20_19-01-02to2019-02-20_19-01-13_1-0007_jpg.rf.14c6af7cc823485d0cf90dd152bfdfe3.jpg"
SPECIES_NAME = "Tamu fisheri"
CLASS_LABEL = "fish"


# === Retrieved Context ===
JOINED_CONTEXT = """
Color: pale yellow or cream with brownish spots. Shape: spherical or ovoid body covered in short, moveable spines. Habitat: deep-sea environments, often found in areas with soft sediments. Size: up to 10 centimeters in diameter. Diet: small invertebrates, detritus. Notable feature: body is highly flexible and can change shape.
---
Color: translucent, colorless. Shape: elongated, gelatinous body with a spherical umbrella and long, sticky tentacles. Habitat: warm, shallow waters, typically found in estuaries or bays. Size: up to 3 centimeters in diameter. Diet: small invertebrates, plankton. Notable feature: bioluminescent, able to emit light.
"""

# === Prompt ===
PROMPT = (
    f"You are a marine biology visual expert.\n"
    f"Knowledge Context:\n{JOINED_CONTEXT.strip()}\n"
    f"<image>\n"
    f"USER: Based on the image and the knowledge above, explain why this image likely contains a {CLASS_LABEL}.\n"
    f"ASSISTANT:"
)

# === Load model ===
print("🚀 Loading model and processor in CPU mode...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
model.eval()
print("✅ Model loaded!")

# === Load and preprocess image ===
print(f"🖼️ Loading image: {IMG_PATH}")
image = Image.open(IMG_PATH).convert("RGB").resize((336, 336))
inputs = processor(images=image, text=PROMPT, return_tensors="pt").to("cpu")

# === Generate explanation ===
print("💬 Generating explanation...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# === Final Output ===
print("\n📢 Explanation Output:\n")
print(f"USER: {PROMPT.split('USER: ')[-1]}")
print(f"ASSISTANT: {output}")

