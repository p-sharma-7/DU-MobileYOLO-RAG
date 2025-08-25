import os
import pandas as pd
from PIL import Image

# === CONFIG ===
LABEL_DIR = "runs/detect/brackish_test_finetuned/labels"
IMAGE_DIR = "brackish_dataset/test/images"
OUTPUT_CSV = "runs/detect/brackish_test_finetuned/detections_filtered.csv"
CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"]

results = []

# Check if label folder exists and has .txt files
if not os.path.exists(LABEL_DIR) or len(os.listdir(LABEL_DIR)) == 0:
    print(f"❌ ERROR: No label files found in {LABEL_DIR}")
    exit()

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    # Resolve JPG or PNG
    base_name = label_file.replace(".txt", "")
    img_path = None
    for ext in [".jpg", ".png"]:
        candidate = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        continue

    with Image.open(img_path) as img:
        width, height = img.size

    label_path = os.path.join(LABEL_DIR, label_file)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height
            confidence = float(parts[5])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)

            results.append({
                "filename": os.path.basename(img_path),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
                "confidence": round(confidence, 4),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

# Save CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Filtered CSV saved to: {OUTPUT_CSV}")
print(f"📈 Total detections with confidence >= {CONFIDENCE_THRESHOLD}: {len(results)}")
