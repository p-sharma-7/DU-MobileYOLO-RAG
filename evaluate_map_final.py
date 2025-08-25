# evaluate_map_final.py

from pathlib import Path
import torch
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image

# CONFIGURATION
GT_PATH = Path("brackish_dataset/test/labels")  # ground truth
PRED_PATH = Path("runs/detect/brackish_test_finetuned/labels")  # UPDATED prediction path
IMG_DIR = Path("brackish_dataset/test/images")
NUM_CLASSES = 6
IOU_THRESHOLDS = [0.5, 0.75, 0.9]

# INIT METRIC
metric = MeanAveragePrecision(iou_thresholds=IOU_THRESHOLDS)

def read_yolo_txt(file_path):
    boxes, labels, scores = [], [], []
    if not file_path.exists():
        return boxes, labels, scores
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            score = float(parts[5]) if len(parts) == 6 else 1.0
            boxes.append([cx, cy, w, h])
            labels.append(cls)
            scores.append(score)
    return boxes, labels, scores

def yolo_to_xyxy(boxes, w, h):
    converted = []
    for cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        converted.append([x1, y1, x2, y2])
    return converted

# EVALUATE ALL IMAGES
image_files = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))

for img_path in image_files:
    img_id = img_path.stem
    with Image.open(img_path) as im:
        w, h = im.size

    # GT and Prediction
    gt_boxes, gt_labels, _ = read_yolo_txt(GT_PATH / f"{img_id}.txt")
    gt_boxes = yolo_to_xyxy(gt_boxes, w, h)

    pred_boxes, pred_labels, pred_scores = read_yolo_txt(PRED_PATH / f"{img_id}.txt")
    pred_boxes = yolo_to_xyxy(pred_boxes, w, h)

    target = [{
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.tensor(gt_labels, dtype=torch.int64)
    }]
    preds = [{
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
        "labels": torch.tensor(pred_labels, dtype=torch.int64),
        "scores": torch.tensor(pred_scores, dtype=torch.float32)
    }]

    metric.update(preds, target)

# FINAL RESULTS
results = metric.compute()
print("\n📊 Final mAP Evaluation Results (Fine-tuned Model):\n")
for k, v in results.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.item():.4f}" if v.ndim == 0 else f"{k}: {[round(i, 4) for i in v.tolist()]}")
    else:
        print(f"{k}: {v}")
