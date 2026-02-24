import os
import sys
import subprocess
import time
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from rfdetr import RFDETRBase

SERVICES = ["firefox", "chrome", "code", "spotify", "libreoffice"] # close all applications
WARMUP_DURATION = 300 # 5min
DISPLAY_BRIGHTNESS = 50 #percent
SLEEP_TIME = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017") # Download dataset at http://images.cocodataset.org/zips/val2017.zip
ENERGY_DIR  = Path(BASE_DIR) / "results"
ENERGIBRIDGE_PATH = "/home/andriana/EnergiBridge/target/release/energibridge" # Change this to the path where Energibridge is installed

# SYSTEM SETUP
def system_setup():
    print("[SETUP] Killing unnecessary services")
    for proc in SERVICES:
        subprocess.run(["pkill", "-f", proc], check=False)

    print("[SETUP] Disabling Wi-Fi...")
    subprocess.run(["sudo", "nmcli", "radio", "wifi", "off"])

    print("[SETUP] Disabling Bluetooth...")
    subprocess.run(["sudo","rfkill", "block", "bluetooth"], check=False)

    print("[SETUP] Setting display brightness...")
    backlight_paths = list(Path("/sys/class/backlight").glob("*/brightness"))
    for path in backlight_paths:
        max_path = path.parent / "max_brightness"
        max_brightness = int(max_path.read_text().strip())
        value = int(max_brightness * DISPLAY_BRIGHTNESS / 100)
        subprocess.run(
            f"echo {value} | sudo tee {path}",
            shell=True,
            check=False,
            capture_output=True
        )

    print("[SETUP] Disabling notification...")
    subprocess.run([ 
        "gsettings", "set", 
        "org.gnome.desktop.notifications", "show-banners", "false"], check=False 
        )
    
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

# WARM UP - GPU Run fibonacci for 5min 

'''  
# LOAD MODELS
def load_yolo():
    print("[MODELS] Loading YOLOv8m...")
    yolo = YOLO("yolov8m.pt")
    print("[MODELS] YOLOv8m Loaded")
    return yolo

def load_rfdetr():
    print("[MODELS] Loading RF-DETR Base...")
    rfdetr = RFDETRBase()
    rfdetr.optimize_for_inference()
    rfdetr.to("cuba")
    print("[MODELS] RF-DETR Loaded")
    return rfdetr

# RUN MODELS
def run_yolo(model):
    print("[INFERENCE] Running YOLOv8m...")
    results = model.predict(source=IMAGES_PATH, save=False, verbose=False, stream=True)
    for _ in results:
        pass
    print("[INFERENCE] YOLOv8m complete.")

def run_rfdetr(model):
    print("[INFERENCE] Running RF-DETR Base...")
    images = list(Path(IMAGES_PATH).glob("*.jpg"))
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        model.predict(img)
    print("[INFERENCE] RF-DETR Base complete.")

# RUN WITH EnergiBridge
def run_experiment(): 
    ENERGY_DIR.mkdir(exist_ok=True) 
    runs = ["yolo"] * 30 + ["rfdetr"] * 30 
    random.shuffle(runs) 
    print(f"[EXPERIMENT] Run order: {runs}\n") 

    for i, model_name in enumerate(runs, 1): 
        if i > 1: 
            print(f"[SLEEP] Waiting {SLEEP_TIME}s...") 
            time.sleep(SLEEP_TIME) 
        
        output = ENERGY_DIR / f"run_{i:02d}_{model_name}.csv" 
        print(f"[RUN {i:02d}/60] model={model_name} results={output.name}") 
        
        subprocess.run([ 
            "sudo", 
            ENERGIBRIDGE_PATH, 
            "--output", str(output), 
            "--summary", "--", 
            "/home/andriana/Documents/Sustainable_Labs/Sustainable_SE/venv/bin/python3", 
            __file__, 
            "--infer", 
            model_name ], check=True) 
    print(f"\n[DONE] All 60 runs complete. Results in: {ENERGY_DIR}")
'''
 # def main():
    
#   if "--infer" in sys.argv:
#        model_name = sys.argv[-1]
#       if model_name == "yolo":
#           run_yolo(load_yolo())
#        elif model_name == "rfdetr":
#           run_rfdetr(load_rfdetr())
#    else:
#        system_setup()
#        cpu_warmup(WARMUP_DURATION)
 #       run_experiment()


if __name__ == "__main__":
    #main()
    system_setup()
    cpu_warmup(WARMUP_DURATION)
