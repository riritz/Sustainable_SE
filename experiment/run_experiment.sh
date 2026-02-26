#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

if [ -z "$ENERGIBRIDGE_PATH" ]; then
    echo "ENERGIBRIDGE_PATH is not set, using default..."
    ENERGIBRIDGE="$HOME/EnergiBridge/target/release/energibridge"
else
    ENERGIBRIDGE="$ENERGIBRIDGE_PATH"
fi

# Create results folder if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "[WARMUP] Running warmup..."
python "$SCRIPT_DIR/warmup.py"
echo "[WARMUP] COMPLETE"


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