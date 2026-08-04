# # import os
# # import random

# # # --- Configuration ---
# # # The local path where your images are currently stored
# # image_dir = '/work/mech-ai-scratch/anneka/food/yolo_produce/obj_train_data/images' 
# # # The prefix path required for the YOLO .txt files
# # path_prefix = '/work/mech-ai-scratch/anneka/food/yolo_produce/obj_train_data/labels'
# # # Supported image extensions
# # valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# # # Split ratios (must sum to 1.0)
# # train_ratio = 0.7
# # val_ratio = 0.15
# # test_ratio = 0.15
# # random.seed(42)

# # def generate_yolo_paths():
# #     # 1. Get all image filenames
# #     files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
# #     if not files:
# #         print(f"No images found in {image_dir}")
# #         return

# #     # 2. Shuffle to ensure random distribution
# #     random.seed(42) # For reproducibility
# #     random.shuffle(files)

# #     # 3. Calculate split indices
# #     total = len(files)
# #     train_end = int(total * train_ratio)
# #     val_end = train_end + int(total * val_ratio)

# #     splits = {
# #         'train.txt': files[:train_end],
# #         'val.txt': files[train_end:val_end],
# #         'test.txt': files[val_end:]
# #     }

# #     # 4. Write to files
# #     for filename, file_list in splits.items():
# #         with open(filename, 'w') as f:
# #             for img in file_list:
# #                 # Combine the prefix with the image name
# #                 # Adjust if your images are in a subfolder like /images/
# #                 full_path = os.path.join(path_prefix, img)
# #                 f.write(full_path + '\n')
        
# #         print(f"Created {filename} with {len(file_list)} images.")

# # if __name__ == "__main__":
# #     generate_yolo_paths()
# import os
# import random

# # CONFIGURATION
# base_dir = '/work/mech-ai-scratch/anneka/food/yolo_produce/obj_train_data/images'
# valid_extensions = ('.jpg', '.jpeg', '.png')

# def generate_yolo_lists():
#     # Get all image files
#     files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith(valid_extensions)]
    
#     if not files:
#         print("Error: No images found. Check your base_dir path.")
#         return

#     random.seed(42)
#     random.shuffle(files)

#     # Ratios
#     train_idx = int(len(files) * 0.8)
#     val_idx = train_idx + int(len(files) * 0.1)

#     splits = {
#         'train.txt': files[:train_idx],
#         'val.txt': files[train_idx:val_idx],
#         'test.txt': files[val_idx:]
#     }

#     for name, content in splits.items():
#         with open(f'/work/mech-ai-scratch/anneka/food/yolo_produce/{name}', 'w') as f:
#             f.write('\n'.join(content) + '\n')
#         print(f"Created {name} with {len(content)} entries.")

# if __name__ == "__main__":
#     generate_yolo_lists()

import os

# The absolute directory where your .txt files are located
target_dir = '/work/mech-ai-scratch/anneka/food/yolo_produce'
files_to_fix = ['train.txt', 'val.txt', 'test.txt']

for filename in files_to_fix:
    full_path = os.path.join(target_dir, filename)
    
    if os.path.exists(full_path):
        # 1. Read the current content
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # 2. Replace 'YOLO_base' with 'yolo_produce'
        # Also ensuring we point to /images/ instead of /labels/
        new_lines = []
        for line in lines:
            new_line = line.replace('YOLO_base', 'yolo_produce')
            new_line = new_line.replace('/labels/', '/images/')
            new_lines.append(new_line)
        
        # 3. Write back the fixed content
        with open(full_path, 'w') as f:
            f.writelines(new_lines)
            
        print(f"DONE: Updated {full_path}")
    else:
        print(f"ERROR: Could not find {full_path}")

# 4. Attempt to delete the cache automatically
cache_path = '/work/mech-ai-scratch/anneka/food/yolo_produce/obj_train_data/labels.cache'
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"CLEANED: Deleted old cache at {cache_path}")