#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENERGIBRIDGE="$HOME/EnergiBridge/target/release/energibridge"
RESULTS_DIR="$SCRIPT_DIR/results"
#ENERGY_IMAGES="$SCRIPT_DIR/dataset/val2017_energy"
#ENERGY_ANN="$SCRIPT_DIR/dataset/annotations_trainval2017/annotations/instances_val2017_energy.json"

# Create results folder if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "[WEIGHTS] Downloading model weights..."
python -c "
from rfdetr import RFDETRNano
from ultralytics import YOLO
RFDETRNano()
YOLO('yolov8m.pt')
"
echo "[WEIGHTS] Downloaded..."

echo "[PROCESSING] Processing dataset..."
python "$SCRIPT_DIR/prepare_dataset.py"

echo "[SETUP] Running setup and warmup..."
python "$SCRIPT_DIR/setup.py"

: '
echo "[ENERGY] Running YOLOv8..."
"$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/yolo_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/exampleyolo.py" \
    --images_dir "$ENERGY_IMAGES" \
    --annotations "$ENERGY_ANN" \
    --model_name medium

echo "[SLEEP] Cooling down for 60s..."
sleep 60

echo "[ENERGY] Running RF-DETR..."
"$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/rfdet_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/examplerfdet.py" \
    --images_dir "$ENERGY_IMAGES" \
    --annotations "$ENERGY_ANN" \
    --model_name nano
'

echo "[ENERGY] Running YOLOv8..."
"$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/yolo_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/yolo_model.py"

echo "[SLEEP] Cooling down for 60s..."
sleep 60

echo "[ENERGY] Running RF-DETR..."
"$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/rfdet_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/rfdet_model.py"

echo "[DONE] Results saved to $RESULTS_DIR"