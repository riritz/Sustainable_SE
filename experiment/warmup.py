import os
import time
from pathlib import Path
from yolo_model import yolo
from rfdet_model import rfdetr

WARMUP_DURATION = 300 # 5min

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017_warmup")

# WARM UP

# WARM UP - CPU Run fibonacci for 5min 
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def cpu_warmup(WARMUP_DURATION):
    print("[WARMUP] Running fibonacci for 5 min...")
    start = time.time()
    while time.time() - start < WARMUP_DURATION:
        fibonacci(10000)
    print("[WARMUP] COMPLETE")

# WARM UP - GPU Run interference on YOLO8vm for 5min 
def gpu_warmup(WARMUP_DURATION):
    print("[WARMUP] GPU warm-up with YOLOv8 inference")

    start = time.time()
    while time.time() - start < WARMUP_DURATION:
        yolo(image_dir=IMAGES_PATH)
        rfdetr(image_dir=IMAGES_PATH)

    print("[WARMUP] GPU warm-up COMPLETE")


if __name__ == "__main__":
    cpu_warmup(WARMUP_DURATION)
    gpu_warmup(WARMUP_DURATION)
