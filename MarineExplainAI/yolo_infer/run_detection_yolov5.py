import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

CLASS_NAMES = ['crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish']

def run_inference(model_path, source, output_dir, conf_thres=0.5, iou_thres=0.65, img_size=640, device='0'):
    device = select_device(device)
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    dataset = LoadImages(source, img_size=img_size, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names

    output_txt_dir = Path(output_dir) / "labels"
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    detections = []

    for path, img, im0s, vid_cap in tqdm(dataset, desc="Running Inference"):
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                h_img, w_img = im0s.shape[:2]

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                    cls_id = int(cls.item())
                    confidence = float(conf.item())

                    cx = ((x1 + x2) / 2) / w_img
                    cy = ((y1 + y2) / 2) / h_img
                    bw = (x2 - x1) / w_img
                    bh = (y2 - y1) / h_img

                    image_name = Path(path).name
                    detections.append({
                        "image": image_name,
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id],
                        "confidence": confidence,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })

                    txt_name = output_txt_dir / f"{Path(path).stem}.txt"
                    with open(txt_name, 'a') as f:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {confidence:.4f}\n")

    pd.DataFrame(detections).to_csv(Path(output_dir) / "detections.csv", index=False)
    print(f"\n✅ Inference complete. {len(detections)} results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv5 model .pt")
    parser.add_argument("--source", type=str, required=True, help="Folder of input images")
    parser.add_argument("--out", type=str, default="MarineExplainAI/results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.out, args.conf, args.iou, args.img_size, args.device)






