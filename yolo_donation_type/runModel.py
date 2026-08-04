from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.train(
    data='/work/mech-ai-scratch/anneka/food/YOLO_base/data.yaml',
    epochs=50,
    imgsz=320,
    batch=2,           # Drop batch size significantly to save RAM
    workers=0,         # Keeps everything on the main process
    # plots=False      # Stops the heavy image generation
    save=True,
    device=0,
    amp=False,         # Turning off AMP can sometimes stop the NMS timeout during setup
    cache=False,        # Ensure we aren't trying to load the whole dataset into RAM
    box=4.0,                     # Stronger box loss for localization
    cls=4.0,  

#     # ✅ Regularization & Dropout
    # dropout=0.2, 
)