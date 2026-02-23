import os
from pathlib import Path
from PIL import Image
from rfdetr import RFDETRNano


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017") # Download dataset at http://images.cocodataset.org/zips/val2017.zip

def rfdetr():
    model = RFDETRNano()
    print("[INFERENCE] Running RF-DETR Base...")
    images = list(Path(IMAGES_PATH).glob("*.jpg"))
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        model.predict(img)
    print("[INFERENCE] RF-DETR Base complete.")

if __name__ == "__main__":
    rfdetr()