import os
import torch
from ultralytics import YOLO

# --- AUTOMATIC PATH SETUP ---
# This ensures the script finds your files regardless of which folder you are in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, "data", "data.yaml")

def train_neuroscan():
    # 1. Hardware Detection
    # Using '0' for your NVIDIA GPU. If for some reason CUDA is not installed, it falls back to CPU.
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*50)
    print("🧠 NEUROSCAN AI: GPU TRAINING MODE")
    print(f"📍 Project Root: {PROJECT_ROOT}")
    print(f"💻 Hardware: {'NVIDIA GPU (Detected)' if device == 0 else 'CPU (Slow Mode)'}")
    print("="*50 + "\n")

    # 2. Load the YOLOv8 'Small' Model
    # 's' is better than 'n' for medical imaging and fits perfectly in 6GB VRAM.
    model = YOLO('yolov8s.pt') 

    # 3. Start the Training
    model.train(
        data=DATA_YAML_PATH,    # Path to your class names and image locations
        epochs=100,               # 50 is the gold standard for high accuracy
        imgsz=640,               # Native resolution for YOLOv8
        batch=-1,                # 16 images per batch (Safe for 6GB VRAM)
        device=device,           # Uses your dedicated GPU
        workers=8,               # Optimized for your i5-13420H cores
        project=os.path.join(PROJECT_ROOT, 'results'), 
        name='brain_tumor_run', 
        exist_ok=True,           # Overwrites the old 'brain_tumor_run' folder
        save=True,                # Saves the best.pt file
        plots=True,               # Generates graphs for your project report
        amp=True,                # Uses Mixed Precision to speed up training
        lr0=0.01,                # Initial learning rate
        patience=10              # Stops early if the model stops improving
    )

    print("\n" + "="*50)
    print("✅ TRAINING SUCCESSFULLY COMPLETED!")
    print(f"🏆 Best Model: {PROJECT_ROOT}/results/brain_tumor_run/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    train_neuroscan()