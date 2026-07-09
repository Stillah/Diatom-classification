import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
from config import *


# ============================
# FUNCTIONS
# ============================

def parse_voc_annotation(xml_path):
    """Parse a VOC XML file and return list of (class_name, bbox)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall(".//object"):   # searches recursively anywhere in the tree:
        name = obj.find("name").text
        bbox = obj.find("bbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return objects

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert VOC bbox (xmin, ymin, xmax, ymax) to YOLO normalized format."""
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    # Clip to [0,1] to avoid errors
    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    width = min(max(width, 0), 1)
    height = min(max(height, 0), 1)
    return x_center, y_center, width, height

def create_yolo_dataset():
    # Locate images and XML annotations
    # Assume images are in DATASET_ROOT/images/ and annotations in DATASET_ROOT/annotations/
    # Adjust if your folder structure is different (e.g., all files in one folder)
    img_dir = DATASET_ROOT / "images"
    ann_dir = DATASET_ROOT / "xmls"

    if not img_dir.exists() or not ann_dir.exists():
        print("Error: Could not find images/ or xmls/ folders.")
        print("Please ensure the dataset is extracted with the following structure:")
        print(f"{DATASET_ROOT}/")
        print("    images/       (all image files)")
        print("    xmls/  (all XML files)")
        return

    # Gather all image files (support jpg, png, jpeg)
    img_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f.name for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in img_extensions]
    
    if not image_files:
        print("No images found. Check the image folder path.")
        return

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(OUTPUT_ROOT / split / "images", exist_ok=True)
        os.makedirs(OUTPUT_ROOT / split / "labels", exist_ok=True)

    # For each image, parse its XML and filter target classes
    valid_samples = []  # (image_path, label_lines)
    skipped_no_annotation = 0
    skipped_no_target = 0

    for img_filename  in image_files:
        # Corresponding XML file (same name, .xml)
        img_path = img_dir / img_filename
        xml_name = Path(img_filename).stem + ".xml"
        xml_path = ann_dir / xml_name
        if not os.path.exists(xml_path):
            skipped_no_annotation += 1
            continue

        # Parse XML
        objects = parse_voc_annotation(xml_path)
        if not objects:
            skipped_no_target += 1
            continue

        # Filter only target classes
        label_lines = []
        for class_name, bbox in objects:
            if class_name not in class_to_id:
                continue
            class_id = class_to_id[class_name]
            # Get image dimensions (needed for normalization)
            # Read the image to get width/height
            img_full_path = img_dir / img_filename
            # Use OpenCV or PIL – here we use OpenCV
            
            img = cv2.imread(str(img_full_path))
            if img is None:
                print(f"Warning: Cannot read image {img_full_path}. Skipping.")
                continue
            h, w = img.shape[:2]
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, w, h)
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if label_lines:
            valid_samples.append((img_filename, label_lines))
        else:
            skipped_no_target += 1

    print(f"Total images: {len(image_files)}")
    print(f"Skipped (no annotation): {skipped_no_annotation}")
    print(f"Skipped (no target class): {skipped_no_target}")
    print(f"Valid samples: {len(valid_samples)}")

    # Split into train/val/test (80:10:10)
    train_val, test = train_test_split(valid_samples, test_size=0.1, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=SEED)  # 0.1111 of 0.9 = 0.1 total

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    # Copy images and write label files
    for split_name, samples in splits.items():
        for img_file, label_lines in samples:
            src_img = img_dir / img_file
            dst_img = OUTPUT_ROOT / split_name / "images" / img_file
            shutil.copy2(src_img, dst_img)

            # Write label file
            label_file = Path(img_file).stem + ".txt"
            dst_label = OUTPUT_ROOT / split_name / "labels" / label_file
            with open(dst_label, "w") as f:
                f.write("\n".join(label_lines))

    # Create data.yaml
    yaml_content = f"""
    # Diatom dataset for YOLO
    path: {os.path.abspath(OUTPUT_ROOT)}
    train: train/images
    val: val/images
    test: test/images

    nc: {len(TARGET_CLASSES)}
    names: {TARGET_CLASSES}
    """
    
    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ YOLO dataset created at: {OUTPUT_ROOT}")
    print(f"   Classes: {TARGET_CLASSES}")
    print(f"   Splits: train={len(train)}, val={len(val)}, test={len(test)}")

# ============================
# RUN
# ============================
if __name__ == "__main__":
    create_yolo_dataset()