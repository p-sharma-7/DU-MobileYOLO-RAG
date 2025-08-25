import pandas as pd
import os

PROMPT_CSV = "results/prompts.csv"  # ✅ Corrected relative path
UPDATED_CSV = "results/prompts_fixed.csv"
CROP_DIR = "MarineExplainAI/crops"
IMG_DIR = "brackish_dataset/test/images"

df = pd.read_csv(PROMPT_CSV)

def update_paths(row):
    row["crop_path"] = os.path.join(CROP_DIR, row["class_name"], row["crop_path"])
    row["original_image"] = os.path.join(IMG_DIR, row["original_image"])
    return row

df = df.apply(update_paths, axis=1)
df.to_csv(UPDATED_CSV, index=False)

print(f"✅ Updated prompts saved to: {UPDATED_CSV}")
