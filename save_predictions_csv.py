import os
import pandas as pd
from PIL import Image

LABEL_DIR = "runs/detect/brackish_test_infer/labels"
IMAGE_DIR = "brackish_dataset/test/images"
OUTPUT_CSV = "runs/detect/brackish_test_infer/detections.csv"

CLASS_NAMES = ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"]

results = []

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    img_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, label_file)

    if not os.path.exists(img_path):
        continue

    with Image.open(img_path) as img:
        width, height = img.size

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                continue  # skip malformed lines

            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            confidence = float(parts[5]) if len(parts) == 6 else None

            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)

            results.append({
                "filename": img_name,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
                "confidence": round(confidence, 3) if confidence else None,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV saved to {OUTPUT_CSV}")
