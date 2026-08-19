import os
import shutil
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMG = os.path.join("assets", "sample_colony_plate.jpg")

def clean_old_dataset():
    """Removes all synthetic images and labels from training_dataset/ and input_images/"""
    dirs_to_clean = [
        "training_dataset/images/train",
        "training_dataset/images/val",
        "training_dataset/labels/train",
        "training_dataset/labels/val",
        "input_images",
        "annotated_outputs"
    ]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("Cleaned out all old dataset directories.")

def process_and_annotate(source_img=None):
    source_img = source_img or DEFAULT_IMG
    if not os.path.exists(source_img):
        raise FileNotFoundError(f"Source image not found at '{source_img}'. Please provide a valid --image-path.")

    # Copy image to training_dataset/images/train/colony_plate_01.jpg and input_images/colony_plate_01.jpg
    dest_train_img = "training_dataset/images/train/colony_plate_01.jpg"
    dest_unlabeled_img = "input_images/colony_plate_01.jpg"
    shutil.copy(source_img, dest_train_img)
    shutil.copy(source_img, dest_unlabeled_img)

    # Read image with OpenCV
    img = cv2.imread(dest_train_img)
    h, w = img.shape[:2]

    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding to isolate white/light colonies on dark brown agar dish
    # Otsu thresholding + adaptive thresholding
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)

    # Morphological operations to separate close colonies
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    box_data = []

    # Dish inner bounding area filter (to ignore outer frame border)
    min_x, max_x = int(w * 0.20), int(w * 0.85)
    min_y, max_y = int(h * 0.12), int(h * 0.90)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 12000: # Filter noise and huge outer border
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        xc_pixel = x + bw / 2.0
        yc_pixel = y + bh / 2.0

        # Must be inside the agar dish region
        if not (min_x <= xc_pixel <= max_x and min_y <= yc_pixel <= max_y):
            continue

        # Compute metric features: Circularity & Extent & Solidity
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-5)
        aspect_ratio = float(bw) / bh if bh > 0 else 1.0

        # Class decision:
        # 0 = small_dot (compact round smooth colony)
        # 1 = sparkling (filamentous, irregular border, starburst, or large fuzzy colony)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-5)

        if area > 450 or circularity < 0.62 or solidity < 0.82 or aspect_ratio > 1.6 or aspect_ratio < 0.6:
            cls_id = 1 # sparkling / fuzzy
        else:
            cls_id = 0 # small_dot

        # Convert to normalized YOLO format
        xc_norm = xc_pixel / float(w)
        yc_norm = yc_pixel / float(h)
        w_norm = bw / float(w)
        h_norm = bh / float(h)

        yolo_lines.append(f"{cls_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        box_data.append((cls_id, x, y, x + bw, y + bh))

    # Write YOLO annotation file: training_dataset/labels/train/colony_plate_01.txt
    txt_dest = "training_dataset/labels/train/colony_plate_01.txt"
    with open(txt_dest, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"Generated {len(yolo_lines)} annotations in '{txt_dest}'")

    # Generate high-res visual preview image
    pil_img = Image.open(dest_train_img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    # Color coding: Class 0 (small_dot) -> Bright Green, Class 1 (sparkling) -> Magenta/Pink
    colors = {0: (0, 255, 100), 1: (255, 0, 150)}
    names = {0: "0:small_dot", 1: "1:sparkling"}

    counts = {0: 0, 1: 0}

    for cls_id, x1, y1, x2, y2 in box_data:
        color = colors[cls_id]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        counts[cls_id] += 1

    # Draw summary overlay banner at the top
    banner_text = f"Annotated Colony Dish | Class 0 (Small Dots): {counts[0]} | Class 1 (Sparklings): {counts[1]}"
    draw.rectangle([0, 0, w, 40], fill=(20, 20, 20))
    draw.text((15, 10), banner_text, fill=(255, 255, 255))

    preview_path = os.path.join("assets", "dataset_annotated_colony_plate.png")
    pil_img.save(preview_path)
    print(f"Saved visual annotation preview to '{preview_path}'")
    print(f"Colony Summary -> Small Dots (0): {counts[0]} | Sparklings (1): {counts[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-annotate colony plate images.")
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMG, help="Path to input colony dish image")
    args = parser.parse_args()

    clean_old_dataset()
    process_and_annotate(source_img=args.image_path)
