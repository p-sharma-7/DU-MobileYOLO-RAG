import os
import pandas as pd
from PIL import Image
from pathlib import Path

# CONFIGURATION
CSV_PATH = "runs/detect/brackish_test_finetuned/detections_filtered.csv"
IMAGE_DIR = "brackish_dataset/test/images"
CROP_DIR = "MarineExplainAI/crops"
CROP_META_CSV = "MarineExplainAI/results/crop_metadata.csv"

os.makedirs(CROP_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
crop_records = []

for idx, row in df.iterrows():
    image_name = row["filename"]
    class_id = row["class_id"]
    class_name = row["class_name"]
    conf = row["confidence"]
    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    full_img_path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(full_img_path):
        continue

    try:
        with Image.open(full_img_path) as img:
            crop = img.crop((xmin, ymin, xmax, ymax))
            crop_filename = f"{Path(image_name).stem}_{class_name}_{idx}.jpg"
            crop_class_dir = os.path.join(CROP_DIR, class_name)
            os.makedirs(crop_class_dir, exist_ok=True)

            crop_path = os.path.join(crop_class_dir, crop_filename)
            crop.save(crop_path)

            crop_records.append({
                "crop_path": crop_path,
                "original_image": image_name,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(conf, 4)
            })

    except Exception as e:
        print(f"❌ Error processing {image_name} - {e}")

# Save all crop metadata
crop_df = pd.DataFrame(crop_records)
crop_df.to_csv(CROP_META_CSV, index=False)

print(f"\n✅ All crops saved in: {CROP_DIR}")
print(f"✅ Crop metadata CSV saved at: {CROP_META_CSV}")
