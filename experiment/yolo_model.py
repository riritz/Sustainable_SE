import os
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017_energy") # Download dataset at http://images.cocodataset.org/zips/val2017.zip
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8m.pt")

def yolo(image_dir: str):
    print("[MODELS] Loading YOLOv8m...")
    model = YOLO(WEIGHTS_PATH)
    print("[MODELS] YOLOv8m Loaded")
    
    images = list(Path(image_dir).glob("*.jpg"))
    print("[INFERENCE] Running YOLOv8m...")
    for img_path in images:
        model.predict(source=str(img_path), batch=1, save=False, verbose=False)


    print("[INFERENCE] YOLOv8m complete.")

if __name__ == "__main__":
    yolo(image_dir=IMAGES_PATH)