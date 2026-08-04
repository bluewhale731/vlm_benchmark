import os

# 1. Configuration
# Path to your existing YOLO labels
input_folder = 'obj_train_data/labels' 
output_folder = 'obj_train_data/labels_cleaned'

os.makedirs(output_folder, exist_ok=True)

# Define the mapping logic:
# We skip: 0 (fresh), 1 (edible_soon), 2 (spoiled)
# We map: 3->0, 4->1, 5->2, 6->3, 7->4, 8->5
mapping = {
    '3': '0', # mold
    '4': '1', # visible_cut
    '5': '2', # leaking
    '6': '3', # discoloration
    '7': '4', # bruising
    '8': '5'  # wrinkling
}

files_processed = 0

# 2. Process Files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        cleaned_lines = []
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                
                # Check if this class is one we want to keep and remap
                if class_id in mapping:
                    new_class_id = mapping[class_id]
                    # Reconstruct the line with the new class ID
                    new_line = f"{new_class_id} " + " ".join(parts[1:])
                    cleaned_lines.append(new_line)
        
        # 3. Save the cleaned file (only if it has remaining labels)
        if cleaned_lines:
            with open(output_path, 'w') as f_out:
                f_out.write("\n".join(cleaned_lines))
            files_processed += 1

print(f"Done! Processed {files_processed} label files.")
print(f"Cleaned labels are in: {output_folder}")