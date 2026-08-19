import sys
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from src.model import get_detection_model
except ImportError:
    from model import get_detection_model

def export_yolo_annotation(boxes, labels, img_size, output_txt_path):
    """
    Exports bounding boxes and class labels in normalized YOLO format:
    class_id x_center y_center width height
    """
    w, h = img_size
    lines = []
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h
        lines.append(f"{int(label)} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

def auto_label_images(
    input_dir,
    output_dir,
    model_path="best_model.pth",
    num_classes=3,
    conf_thresh=0.4,
    class_names=None
):
    """
    Automated annotation extraction pipeline:
    Takes unlabeled images, runs inference, and writes annotation files & visual previews.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = class_names or {0: "background", 1: "small_dot", 2: "sparkling"}


    # Load model
    if os.path.exists(model_path):
        print(f"Loading trained model checkpoint from '{model_path}'...")
        checkpoint = torch.load(model_path, map_location=device)
        backbone = checkpoint.get("backbone", "mobilenet")
        num_classes = checkpoint.get("num_classes", num_classes)
        model = get_detection_model(num_classes=num_classes, pretrained=False, backbone_type=backbone)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"No checkpoint found at '{model_path}'. Initializing pretrained COCO model...")
        model = get_detection_model(num_classes=num_classes, pretrained=True, backbone_type="mobilenet")

    model.to(device)
    model.eval()

    labels_out_dir = os.path.join(output_dir, "labels")
    previews_out_dir = os.path.join(output_dir, "annotated_images")
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(previews_out_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"Extracting annotations for {len(image_files)} images in '{input_dir}'...")

    total_boxes_extracted = 0

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size

            # Prepare tensor
            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            predictions = model(img_tensor)[0]

            boxes = predictions["boxes"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()

            # Filter by confidence threshold
            keep_indices = np.where(scores >= conf_thresh)[0]
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            filtered_labels = labels[keep_indices]

            total_boxes_extracted += len(filtered_boxes)

            # Export YOLO txt (Combined)
            base_name = os.path.splitext(img_name)[0]
            txt_out_path = os.path.join(labels_out_dir, f"{base_name}.txt")
            export_yolo_annotation(filtered_boxes, filtered_labels, (w, h), txt_out_path)

            # Export Separate Class txt files
            small_dots_dir = os.path.join(output_dir, "labels_small_dots")
            sparklings_dir = os.path.join(output_dir, "labels_sparklings")
            os.makedirs(small_dots_dir, exist_ok=True)
            os.makedirs(sparklings_dir, exist_ok=True)

            dots_mask = (filtered_labels == 1) | (filtered_labels == 0) # Support 0/1 indexing
            dots_boxes = filtered_boxes[dots_mask]
            dots_lbls = filtered_labels[dots_mask]
            export_yolo_annotation(dots_boxes, dots_lbls, (w, h), os.path.join(small_dots_dir, f"{base_name}_dots.txt"))

            sparkle_boxes = filtered_boxes[filtered_labels == 2]
            sparkle_lbls = filtered_labels[filtered_labels == 2]
            export_yolo_annotation(sparkle_boxes, sparkle_lbls, (w, h), os.path.join(sparklings_dir, f"{base_name}_sparklings.txt"))

            # Palette: Class 1 (small_dot -> Green), Class 2 (sparkling -> Magenta/Orange)
            color_map = {1: (50, 220, 50), 2: (255, 50, 200), 0: (50, 150, 255)}

            # 1. Draw Combined Visual Preview
            draw_img_combined = pil_img.copy()
            draw_comb = ImageDraw.Draw(draw_img_combined)
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                color = color_map.get(int(label), (255, 255, 0))
                draw_comb.rectangle(box, outline=color, width=3)
                lbl_name = class_names.get(int(label), f"class_{label}")
                draw_comb.text((box[0] + 3, box[1] + 3), f"{lbl_name}:{score:.2f}", fill=(255, 255, 255))
            draw_img_combined.save(os.path.join(previews_out_dir, f"{base_name}_annotated.png"))

            # 2. Draw Class 1 (Small Dots) ONLY Preview
            draw_img_dots = pil_img.copy()
            draw_dots = ImageDraw.Draw(draw_img_dots)
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                if label == 1 or label == 0:
                    draw_dots.rectangle(box, outline=(50, 220, 50), width=3)
                    draw_dots.text((box[0] + 3, box[1] + 3), f"small_dot:{score:.2f}", fill=(255, 255, 255))
            
            dots_preview_dir = os.path.join(output_dir, "previews_small_dots")
            os.makedirs(dots_preview_dir, exist_ok=True)
            draw_img_dots.save(os.path.join(dots_preview_dir, f"{base_name}_small_dots.png"))

            # 3. Draw Class 2 (Sparklings) ONLY Preview
            draw_img_spark = pil_img.copy()
            draw_spark = ImageDraw.Draw(draw_img_spark)
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                if label == 2:
                    draw_spark.rectangle(box, outline=(255, 50, 200), width=3)
                    draw_spark.text((box[0] + 3, box[1] + 3), f"sparkling:{score:.2f}", fill=(255, 255, 255))

            sparkle_preview_dir = os.path.join(output_dir, "previews_sparklings")
            os.makedirs(sparkle_preview_dir, exist_ok=True)
            draw_img_spark.save(os.path.join(sparkle_preview_dir, f"{base_name}_sparklings.png"))

            n_dots = len(dots_boxes)
            n_sparkles = len(sparkle_boxes)
            print(f"  [+] {img_name}: Found {n_dots} Small Dots, {n_sparkles} Sparklings (Total {len(filtered_boxes)} colonies)")

    print(f"\nAuto-Labeling Complete!")
    print(f"Annotated Images       -> '{previews_out_dir}/'")
    print(f"Combined Annotations   -> '{labels_out_dir}/'")
    print(f"Small Dots Annotations -> '{small_dots_dir}/'")
    print(f"Sparklings Annotations -> '{sparklings_dir}/'")
    print(f"Small Dots Previews    -> '{dots_preview_dir}/'")
    print(f"Sparklings Previews    -> '{sparkle_preview_dir}/'")

def main():
    parser = argparse.ArgumentParser(description="Auto-labeler & Annotation Extractor engine.")
    parser.add_argument("--input-dir", type=str, default="input_images", help="Directory of unlabeled images")
    parser.add_argument("--output-dir", type=str, default="annotated_outputs", help="Output directory for annotations and previews")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to trained PyTorch model weights")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Minimum confidence threshold for auto-annotation")
    args = parser.parse_args()

    auto_label_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        conf_thresh=args.conf_thresh
    )

if __name__ == "__main__":
    main()
