import os
import torch
import subprocess
import time
from pathlib import Path
#from exampleyolo import process_dataset 
from yolo_model import yolo

SERVICES = ["firefox", "chrome", "code", "spotify", "libreoffice"] # close all applications
WARMUP_DURATION = 300 # 5min
DISPLAY_BRIGHTNESS = 50 #percent
SLEEP_TIME = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset", "val2017_warmup")
IMAGES_ANN_PATH = os.path.join(os.path.join(BASE_DIR, 
                                            "dataset", 
                                            "annotations_trainval2017", 
                                            "annotations", 
                                            "instances_val2017_warmup.json") )

# SYSTEM SETUP
def system_setup():
    print("[SETUP] Killing unnecessary services")
    for proc in SERVICES:
        subprocess.run(["pkill", "-f", proc], check=False)

    print("[SETUP] Disabling Wi-Fi...")
    subprocess.run(["sudo", "nmcli", "radio", "wifi", "off"])

    print("[SETUP] Disabling Bluetooth...")
    subprocess.run(["sudo","rfkill", "block", "bluetooth"], check=False)

    print("[SETUP] Disabling auto-brightness...")
    subprocess.run(["sudo", "systemctl", "stop", "iio-sensor-proxy"], check=False)
    subprocess.run(["gsettings", "set",
                    "org.gnome.settings-daemon.plugins.power",
                    "ambient-enabled", "false"], check=False)

    print("[SETUP] Disabling idle dimming...")
    subprocess.run(["gsettings", "set",
                    "org.gnome.settings-daemon.plugins.power",
                    "idle-dim", "false"], check=False)

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

# WARM UP - GPU Run interference on YOLO8vm for 5min 
def gpu_warmup(WARMUP_DURATION):
    print("[WARMUP] GPU warm-up with YOLOv8 inference")

    start = time.time()
    while time.time() - start < WARMUP_DURATION:
        #process_dataset(
        #    images_dir=IMAGES_PATH,
        #    annotations_path=IMAGES_ANN_PATH,
        #    model_name="medium"
        #)
        yolo()

    print("[WARMUP] GPU warm-up COMPLETE")


if __name__ == "__main__":
    system_setup()
    cpu_warmup(WARMUP_DURATION)
    gpu_warmup(WARMUP_DURATION)
