import os
import urllib.request

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

downloads = {
    os.path.join(MODEL_DIR, "yolo_person_detection.pt"):
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    os.path.join(MODEL_DIR, "yolo_box_detection.pt"):
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
}

for filepath, url in downloads.items():
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"{os.path.basename(filepath)} already exists. Skipping.")