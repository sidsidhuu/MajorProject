import os
from ultralytics import YOLO

# Dynamic Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data_seg.yaml")

def train():
    # Load Segmentation Model
    model = YOLO('yolov8s-seg.pt') 

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0, # Your RTX 4050
        project=os.path.join(PROJECT_ROOT, 'results'),
        name='brain_tumor_seg',
        amp=True
    )

if __name__ == "__main__":
    train()