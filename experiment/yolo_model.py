import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017") # Download dataset at http://images.cocodataset.org/zips/val2017.zip

# runmodel
def yolo():
    print("[MODELS] Loading YOLOv8m...")
    model = YOLO("yolov8m.pt")
    print("[MODELS] YOLOv8m Loaded")

    print("[INFERENCE] Running YOLOv8m...")
    results = model.predict(source=IMAGES_PATH, save=False, verbose=False, stream=True)
    for _ in results:
        pass
    print("[INFERENCE] YOLOv8m complete.")

if __name__ == "__main__":
    yolo()