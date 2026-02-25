import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017_energy") # Download dataset at http://images.cocodataset.org/zips/val2017.zip
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8m.pt")

def yolo(image_dir):
    print("[MODELS] Loading YOLOv8m...")
    model = YOLO(WEIGHTS_PATH)
    print("[MODELS] YOLOv8m Loaded")

    print("[INFERENCE] Running YOLOv8m...")
    results = model.predict(source=image_dir, save=False, verbose=False, stream=True)
    for _ in results:
        pass
    print("[INFERENCE] YOLOv8m complete.")

if __name__ == "__main__":
    yolo(image_dir=IMAGES_PATH)