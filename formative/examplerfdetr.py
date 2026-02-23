import argparse
import json
import os
import shutil
from typing import Dict
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def get_model_weights_path(model_name: str) -> str:
    weights_path = os.path.join(WEIGHTS_DIR, f"rf-detr-{model_name}.pth")
    if not os.path.exists(weights_path):
        if model_name == "nano":
            model = RFDETRNano()
        elif model_name == "small":
            model = RFDETRSmall()
        elif model_name == "medium":
            model = RFDETRMedium()
        elif model_name == "large":
            model = RFDETRLarge()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        shutil.move(f"rf-detr-{model_name}.pth", weights_path)
    return weights_path

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def process_dataset(images_dir: str, annotations_path: str, model_name: str) -> Dict:
    device = "cuda"
    threshold = 0.5
    labels = list(range(len(COCO_CLASSES)))

    weights_path = get_model_weights_path(model_name)

    if model_name == "nano":
        model = RFDETRNano(device=device, pretrain_weights=weights_path)
    elif model_name == "small":
        model = RFDETRSmall(device=device, pretrain_weights=weights_path)
    elif model_name == "medium":
        model = RFDETRMedium(device=device, pretrain_weights=weights_path)
    elif model_name == "large":
        model = RFDETRLarge(device=device, pretrain_weights=weights_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    results_list = []
    total_gt = 0
    total_tp = 0
    total_fp = 0

    for img_info in coco_data["images"]:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = os.path.join(images_dir, file_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        img_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        predictions = model.predict(img_rgb, threshold=threshold)

        pred_boxes = np.array(predictions.xyxy) if predictions.xyxy is not None else []
        pred_scores = np.array(predictions.confidence) if predictions.confidence is not None else []
        pred_classes = np.array(predictions.class_id) if predictions.class_id is not None else []

        gt_boxes = [
            [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
            for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]
        total_gt += len(gt_boxes)
        matched_gt = set()

        for xyxy, conf, cls_id in zip(pred_boxes, pred_scores, pred_classes):
            if (cls_id - 1) not in labels:
                continue
            best_iou = 0
            best_idx = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(xyxy, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= 0.5:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

            x1, y1, x2, y2 = map(float, xyxy)
            results_list.append({
                "image_id": image_id,
                "object_id": f"{int(cls_id)}_{image_id}",
                "score": float(conf),
                "label": COCO_CLASSES[int(cls_id)],
                "bbox": [x1, y1, x2, y2]
            })

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_gt + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "results": results_list,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to dataset images")
    parser.add_argument("--annotations", required=True, help="Path to COCO detection JSON annotations")
    parser.add_argument("--model_name", choices=["nano", "small", "medium", "large"], default="medium")
    args = parser.parse_args()

    print("\nStarting submodule: RF-DETR Dataset Object Detection\nProcessing images...\n")

    start_time = time.time()

    output_data = process_dataset(args.images_dir, args.annotations, args.model_name)

    end_time = time.time()
    elapsed_time = end_time - start_time

    output_path = "detections.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Processing complete. JSON output saved to {output_path}\n")
    print("Metrics:")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(json.dumps(output_data["metrics"], indent=2))