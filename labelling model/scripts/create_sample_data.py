import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Class mapping
CLASS_MAP = {
    0: "background",
    1: "small_dot",
    2: "sparkling"
}


def generate_sample_dataset(output_dir="training_dataset", num_train=15, num_val=5, img_size=(400, 400)):
    """
    Generates a synthetic image dataset of geometric shapes with YOLO annotation text files.
    """
    train_img_dir = os.path.join(output_dir, "images", "train")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")
    unlabeled_dir = "input_images"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, unlabeled_dir]:
        os.makedirs(d, exist_ok=True)

    def create_synthetic_image(img_path, lbl_path=None):
        # Canvas background (light grey / off-white)
        bg_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
        img = Image.new("RGB", img_size, color=bg_color)
        draw = ImageDraw.Draw(img)

        w, h = img_size
        num_shapes = random.randint(1, 3)
        annotations = []

        for _ in range(num_shapes):
            shape_type = random.choice([1, 2]) # 1: rectangle, 2: circle
            box_w = random.randint(50, 120)
            box_h = random.randint(50, 120)
            x1 = random.randint(10, w - box_w - 10)
            y1 = random.randint(10, h - box_h - 10)
            x2 = x1 + box_w
            y2 = y1 + box_h

            color = (random.randint(50, 220), random.randint(50, 220), random.randint(50, 220))

            if shape_type == 1:
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

            if lbl_path is not None:
                # Convert to normalized YOLO format: class_id x_center y_center width height
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                annotations.append(f"{shape_type} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        img.save(img_path)

        if lbl_path is not None:
            with open(lbl_path, "w") as f:
                f.write("\n".join(annotations))

    print(f"Generating {num_train} training samples...")
    for i in range(num_train):
        img_p = os.path.join(train_img_dir, f"sample_train_{i:03d}.png")
        lbl_p = os.path.join(train_lbl_dir, f"sample_train_{i:03d}.txt")
        create_synthetic_image(img_p, lbl_p)

    print(f"Generating {num_val} validation samples...")
    for i in range(num_val):
        img_p = os.path.join(val_img_dir, f"sample_val_{i:03d}.png")
        lbl_p = os.path.join(val_lbl_dir, f"sample_val_{i:03d}.txt")
        create_synthetic_image(img_p, lbl_p)

    print("Generating 5 unlabeled test images for auto-labeler testing...")
    for i in range(5):
        img_p = os.path.join(unlabeled_dir, f"unlabeled_{i:03d}.png")
        create_synthetic_image(img_p, lbl_path=None)

    print(f"Dataset successfully created in '{output_dir}/'")

if __name__ == "__main__":
    generate_sample_dataset()
