import argparse
import json
import os
from typing import Dict
import cv2
from ultralytics import YOLO
import numpy as np
import time

MODEL_MAP = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt"
}

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
    model_path = MODEL_MAP[model_name]
    model = YOLO(model_path)

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

        results = model(image)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        scores = results.boxes.conf.cpu().numpy() if results.boxes else []
        class_ids = results.boxes.cls.cpu().numpy() if results.boxes else []

        gt_boxes = [
            [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
            for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]
        total_gt += len(gt_boxes)
        matched_gt = set()

        for xyxy, conf, cls_id in zip(boxes, scores, class_ids):
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
                "label": model.names[int(cls_id)],
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

    print("\nStarting submodule: YOLOv8 Dataset Object Detection\nProcessing images...\n")

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