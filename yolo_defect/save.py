import json
import os
from ultralytics import YOLO

# 1. Load your trained model
# Replace 'path/to/best.pt' with your actual weights path (e.g., 'runs/detect/train/weights/best.pt')
model = YOLO('runs/detect/train23/weights/best.pt')

# 2. Path to your images
IMAGE_DIR = "obj_train_data/images"
output_file = "defect_expert_predictions.json"

expert_data = {}

# 3. Run inference on the folder
# stream=True helps save memory on your 8GB GPU
results = model.predict(source=IMAGE_DIR, conf=0.25, stream=True)

print("Starting YOLO classification...")

for result in results:
    # Get the filename from the path
    filename = os.path.basename(result.path)
    
    # If YOLO detected something
    if len(result.boxes) > 0:
        # Get the class with the highest confidence score
        top_box = result.boxes[0]
        class_id = int(top_box.cls[0])
        label = model.names[class_id]
        confidence = float(top_box.conf[0])
    else:
        label = "unknown"
        confidence = 0.0

    # Store it in our dictionary
    expert_data[filename] = {
        "category": label,
        "yolo_conf": confidence
    }

# 4. Save to JSON
with open(output_file, 'w') as f:
    json.dump(expert_data, f, indent=4)

print(f"Done! Saved predictions for {len(expert_data)} images to {output_file}")