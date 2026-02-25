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
from rfdetr import RFDETRMedium
from ultralytics import YOLO
RFDETRMedium()
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

echo "[ENERGY] Running YOLOv8..."
time "$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/yolo_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/yolo_model.py"

echo "[SLEEP] Cooling down for 30s..."
sleep 30

echo "[ENERGY] Running RF-DETR..."
time "$ENERGIBRIDGE" \
  --output "$RESULTS_DIR/rfdet_run1.csv" \
  --gpu \
  -- \
  python "$SCRIPT_DIR/rfdet_model.py"

echo "[DONE] Results saved to $RESULTS_DIR"
'

echo "[ENERGY] Generating shuffled run order..."
python - <<'EOF'
import random, json

runs = ["yolo"] * 30 + ["rfdetr"] * 30
random.shuffle(runs)

with open("/tmp/run_order.txt", "w") as f:
    for r in runs:
        f.write(r + "\n")

print(f"  Run order: {runs}")
EOF

# Run experiment
yolo_count=0
rfdetr_count=0
run_number=0

while IFS= read -r model; do
    run_number=$((run_number + 1))

    if [ "$model" = "yolo" ]; then
        yolo_count=$((yolo_count + 1))
        label="yolo"
        count=$yolo_count
        script="$SCRIPT_DIR/yolo_model.py"
    else
        rfdetr_count=$((rfdetr_count + 1))
        label="rfdetr"
        count=$rfdetr_count
        script="$SCRIPT_DIR/rfdet_model.py"
    fi

    output_csv="$RESULTS_DIR/${label}_run${count}.csv"

    echo "[RUN $run_number/60] Model=$label  Run#$count"

    echo "[SLEEP] Cooling down for 30s..."
    sleep 30

    time "$ENERGIBRIDGE" \
        --output "$output_csv" \
        --gpu \
        -- \
        python "$script"

    echo "[DONE] Run $run_number complete. Results saved to $output_csv"

done < /tmp/run_order.txt

echo "All 60 runs finished."
echo "Results saved to: $RESULTS_DIR"