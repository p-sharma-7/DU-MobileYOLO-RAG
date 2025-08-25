import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === Monkey patch to disable GPU tracking in CodeCarbon ===
import codecarbon.core.resource_tracker as rt
def no_gpu_tracking(self):
    self.gpu = False
    self.gpu_count = 0
    self._gpu_ids = []
    self._gpu_power_usage = 0
    self._gpu_energy = 0.0
    self._last_gpu_energy = 0.0
    self._last_gpu_time = 0.0
rt.ResourceTracker.set_GPU_tracking = no_gpu_tracking

from codecarbon import EmissionsTracker

# === Set absolute log path ===
tracker = EmissionsTracker(
    project_name="BM25_RAG_Real",
    output_dir="/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/rag/real/logs",
    log_level="info",
    measure_power_secs=1,
)
tracker.start()

# === Paths ===
INPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/prompts_rag_real.csv"
OUTPUT_CSV = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/results/rag_real/explanations_rag_real_sample30.csv"
MODEL_DIR = "/workspace/Marine/DU-MobileYOLO-main/MarineExplainAI/vlm_models/llava_v1_5_7b/checkpoints"

# === Load model & processor ===
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

# === Load 30-sample prompt CSV ===
df = pd.read_csv(INPUT_CSV)
df30 = df.sample(30, random_state=42).reset_index(drop=True)

# === Run generation and save results ===
results = []
for i, row in tqdm(df30.iterrows(), total=len(df30)):
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

    # Incremental save
    if (i + 1) % 5 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

# Final save
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(results)} explanations to {OUTPUT_CSV}")

# === End emissions tracking ===
tracker.stop()
