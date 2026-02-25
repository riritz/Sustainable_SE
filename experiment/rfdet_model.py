import os
from pathlib import Path
from PIL import Image
from rfdetr import RFDETRMedium


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017_energy") # Download dataset at http://images.cocodataset.org/zips/val2017.zip
WEIGHTS_PATH = os.path.join(BASE_DIR, "rf-detr-medium.pth")

def rfdetr():
    model = RFDETRMedium(pretrain_weights=WEIGHTS_PATH)
    print("[INFERENCE] Running RF-DETR Medium...")
    images = list(Path(IMAGES_PATH).glob("*.jpg"))
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        model.predict(img)
    print("[INFERENCE] RF-DETR Nano complete.")

if __name__ == "__main__":
    rfdetr()