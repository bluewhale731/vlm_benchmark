import os

label_to_remove = "5"  # class you want to remove
folder = "obj_train_data/labels"      # path to your txt files

for file in os.listdir(folder):
    if file.endswith(".txt"):
        path = os.path.join(folder, file)
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        # keep only lines NOT starting with that label
        new_lines = [line for line in lines if not line.startswith(label_to_remove + " ")]
        
        with open(path, "w") as f:
            f.writelines(new_lines)