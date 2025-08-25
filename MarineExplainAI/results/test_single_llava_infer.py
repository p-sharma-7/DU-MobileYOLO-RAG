import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# === CONFIGURATION ===
model_path = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"
image_path = "/workspace/Marine/DU-MobileYOLO-main/brackish_dataset/test/images/2019-03-25_23-17-56to2019-03-25_23-18-04_1-0098_jpg.rf.e480a9ad049b2e0567a7b94720f60a6e.jpg"
species = "crab"

# === STEP 1: Load and resize the image ===
try:
    image = Image.open(image_path).convert("RGB").resize((336, 336))
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit()

# === STEP 2: Define the dialog-style prompt ===
prompt = f"<image>\nUSER: What marine species is likely visible in this scene containing a {species}?\nASSISTANT:"

# === STEP 3: Load processor and model ===
print("🚀 Loading LLaVA model...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
print("✅ Model loaded!")

# === STEP 4: Preprocess and move to device ===
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

# === STEP 5: Run inference ===
try:
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n📢 LLaVA Explanation:\n", output)
except Exception as e:
    print(f"❌ Inference failed: {e}")
