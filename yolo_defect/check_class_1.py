import argparse
import os
from collections import Counter

import yaml


def load_class_names(yaml_path):
    """Pull the {id: name} map from a YOLO data.yaml. Returns {} if unavailable."""
    if not yaml_path or not os.path.isfile(yaml_path):
        return {}
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    # names can be a dict {0: 'produce', ...} or a list ['produce', ...]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def count_labels(labels_dir):
    """Walk a folder of YOLO .txt label files and tally per-class stats."""
    instance_counts = Counter()   # total bounding boxes per class
    image_counts = Counter()      # images that contain at least one of a class
    n_files = 0
    n_empty = 0
    n_boxes = 0

    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            # skip the YOLO file lists themselves if they live in this dir
            if fname in ('train.txt', 'val.txt', 'test.txt'):
                continue
            n_files += 1
            classes_in_image = set()
            with open(os.path.join(root, fname), 'r') as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if not lines:
                n_empty += 1
            for ln in lines:
                cls_id = int(float(ln.split()[0]))
                instance_counts[cls_id] += 1
                classes_in_image.add(cls_id)
                n_boxes += 1
            for cls_id in classes_in_image:
                image_counts[cls_id] += 1

    return instance_counts, image_counts, n_files, n_empty, n_boxes


def main():
    parser = argparse.ArgumentParser(description="Count YOLO label instances per class.")
    parser.add_argument('labels_dir', help="Folder containing YOLO .txt label files")
    parser.add_argument('--yaml', default=None,
                        help="Path to data.yaml for class names (optional)")
    args = parser.parse_args()

    names = load_class_names(args.yaml)
    instance_counts, image_counts, n_files, n_empty, n_boxes = count_labels(args.labels_dir)

    print(f"\nScanned {n_files} label files ({n_empty} empty), {n_boxes} total boxes\n")

    all_ids = sorted(set(instance_counts) | set(image_counts) | set(names))
    header = f"{'id':>3}  {'class':<14}{'instances':>11}{'images':>9}"
    print(header)
    print("-" * len(header))
    for cls_id in all_ids:
        name = names.get(cls_id, f"<unknown {cls_id}>")
        print(f"{cls_id:>3}  {name:<14}{instance_counts[cls_id]:>11}{image_counts[cls_id]:>9}")

    # Flag any class ids that appear in labels but not in the yaml
    unknown = [c for c in instance_counts if names and c not in names]
    if unknown:
        print(f"\nWARNING: class ids {sorted(unknown)} found in labels but not in the yaml.")


if __name__ == '__main__':
    main()
    
    
