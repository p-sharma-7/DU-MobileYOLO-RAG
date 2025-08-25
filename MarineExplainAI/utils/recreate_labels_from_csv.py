import os
import pandas as pd
from PIL import Image, ImageDraw

CSV_PATH = "runs/detect/brackish_test_finetuned/detections_filtered.csv"
LABEL_OUT_DIR = "runs/detect/brackish_test_finetuned/labels"
IMG_DIR = "brackish_dataset/test/images"
IMG_OUT_DIR = "runs/detect/brackish_test_finetuned"

os.makedirs(LABEL_OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    filename = row["filename"]
    cls_id = int(row["class_id"])
    conf = float(row["confidence"])
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

    image_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(image_path):
        continue  # Skip missing images

    with Image.open(image_path) as img:
        w, h = img.size

        # Convert to YOLO format
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        label_file = os.path.join(LABEL_OUT_DIR, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(label_file, "a") as f:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")

        # Optional: save visual detection box image again
        save_img_path = os.path.join(IMG_OUT_DIR, filename)
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        img_draw.save(save_img_path)

print(f"\n✅ Recreated {len(df)} detections in YOLO label format under: {LABEL_OUT_DIR}")
