#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[WEIGHTS] Downloading model weights (requires internet connection)..."
python -c "
from rfdetr import RFDETRMedium
from ultralytics import YOLO
RFDETRMedium()
YOLO('yolov8m.pt')
"
echo "[WEIGHTS] Downloaded..."

echo "[PROCESSING] Processing dataset..."
python "$SCRIPT_DIR/prepare_dataset.py"

echo "[PREPARE] COMPLETE"