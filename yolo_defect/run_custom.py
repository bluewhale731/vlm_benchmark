from ultralytics import YOLO
import torch

def main():
    # 1. Double check CUDA inside the script
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
    
    # 2. Load a model (YOLOv8, YOLOv11, or YOLO26 depending on your version)
    # 'n' is nano (fast), 'x' is extra-large (best for A100)
    model = YOLO("yolo11n.pt") 

    # 3. Train the model
    # imgsz: A100 can easily handle 640 or even 1280
    # batch: Start with 32 or 64 for an 80GB A100
    model.train(
        data="data.yaml", 
        epochs=80, 
        cls=2.0,
        imgsz=640, 
        batch=4, 
        device=0,      # Use the first A100
        amp=True       # Enable Mixed Precision for faster A100 training
    )

if __name__ == "__main__":
    main()