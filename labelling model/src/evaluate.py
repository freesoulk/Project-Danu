import sys
from pathlib import Path
import os
import argparse
# pyrefly: ignore [missing-import]
import torch
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from src.dataset import YOLOImageDataset, collate_fn
    from src.model import get_detection_model
except ImportError:
    from dataset import YOLOImageDataset, collate_fn
    from model import get_detection_model

def calculate_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def evaluate_model(val_img_dir, val_lbl_dir, model_path="best_model.pth", conf_thresh=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found. Please train model first.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    backbone = checkpoint.get("backbone", "mobilenet")
    num_classes = checkpoint.get("num_classes", 3)

    model = get_detection_model(num_classes=num_classes, pretrained=False, backbone_type=backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_dataset = YOLOImageDataset(val_img_dir, val_lbl_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out_dir = os.path.join(os.path.dirname(val_img_dir), "..", "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    class_names = {0: "bg", 1: "rectangle", 2: "circle"}
    total_gt = 0
    total_matches = 0
    iou_scores = []

    print(f"Evaluating model on {len(val_dataset)} validation images...\n")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            img_info = val_dataset.img_files[idx]
            pil_img = Image.open(os.path.join(val_img_dir, img_info)).convert("RGB")
            w, h = pil_img.size

            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred = model(img_tensor)[0]

            gt_boxes = targets[0]["boxes"].numpy()
            gt_labels = targets[0]["labels"].numpy()

            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()

            keep = np.where(pred_scores >= conf_thresh)[0]
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            total_gt += len(gt_boxes)

            # Match GT with Pred
            matched_gt = set()
            for p_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                if best_iou >= 0.5 and best_gt_idx != -1:
                    matched_gt.add(best_gt_idx)
                    total_matches += 1
                    iou_scores.append(best_iou)

            # Side-by-side ground truth vs prediction comparison
            comp_img = Image.new("RGB", (w * 2 + 10, h + 30), color=(240, 240, 240))
            comp_img.paste(pil_img, (0, 30))
            comp_img.paste(pil_img, (w + 10, 30))

            draw = ImageDraw.Draw(comp_img)
            draw.text((10, 5), f"Ground Truth (Target: {len(gt_boxes)})", fill=(0, 100, 0))
            draw.text((w + 20, 5), f"Model Predictions (Found: {len(pred_boxes)})", fill=(0, 0, 150))

            # Draw GT (Green)
            for g_box, g_lbl in zip(gt_boxes, gt_labels):
                draw.rectangle([g_box[0], g_box[1] + 30, g_box[2], g_box[3] + 30], outline=(0, 200, 0), width=3)
                draw.text((g_box[0] + 4, g_box[1] + 34), class_names.get(g_lbl, str(g_lbl)), fill=(0, 150, 0))

            # Draw Preds (Blue)
            for p_box, p_score, p_lbl in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = p_box[0] + w + 10, p_box[1] + 30, p_box[2] + w + 10, p_box[3] + 30
                draw.rectangle([x1, y1, x2, y2], outline=(0, 100, 255), width=3)
                tag = f"{class_names.get(p_lbl, str(p_lbl))}:{p_score:.2f}"
                draw.text((x1 + 4, y1 + 4), tag, fill=(0, 50, 200))

            save_path = os.path.join(out_dir, f"eval_comparison_{idx:03d}.png")
            comp_img.save(save_path)

    avg_mIoU = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    recall = (total_matches / float(total_gt)) if total_gt > 0 else 0.0

    print("================ EVALUATION METRICS ================")
    print(f"Total Ground Truth Bounding Boxes : {total_gt}")
    print(f"Total Correctly Matched Boxes     : {total_matches}")
    print(f"Detection Recall Rate             : {recall * 100:.2f}%")
    print(f"Mean IoU (Intersection over Union): {avg_mIoU:.4f}")
    print(f"Visual Comparisons Saved to       : '{out_dir}/'")
    print("====================================================")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Labelling model.")
    parser.add_argument("--data-dir", type=str, default="training_dataset", help="Root data directory")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Checkpoint model file")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Score threshold for predictions")
    args = parser.parse_args()

    val_img_dir = os.path.join(args.data_dir, "images", "val")
    val_lbl_dir = os.path.join(args.data_dir, "labels", "val")

    evaluate_model(val_img_dir, val_lbl_dir, model_path=args.model_path, conf_thresh=args.conf_thresh)

if __name__ == "__main__":
    main()
