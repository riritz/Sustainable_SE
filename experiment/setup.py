import os
import subprocess
from pathlib import Path
from yolo_model import yolo
from rfdet_model import rfdetr

SERVICES = ["firefox", "chrome", "code", "spotify", "libreoffice"] # close all applications
DISPLAY_BRIGHTNESS = 50 #percent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    print("[SETUP] COMPLETE")

if __name__ == "__main__":
    system_setup()