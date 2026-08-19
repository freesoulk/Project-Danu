import os
import json
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset
from PIL import Image
# pyrefly: ignore [missing-import]
import numpy as np

class YOLOImageDataset(Dataset):
    """
    PyTorch Dataset for object detection reading YOLO format annotations (.txt).
    Each annotation line: class_id x_center y_center width height (normalized 0..1).
    """
    def __init__(self, img_dir, label_dir, transforms=None, class_names=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.class_names = class_names or []
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.img_files = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # Map YOLO 0-indexed class (0: small_dot) to PyTorch 1-indexed target class (1: small_dot, 0: background)
                        if cls_id == 0:
                            cls_id = 1
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Convert YOLO normalized (xc, yc, w, h) to absolute (xmin, ymin, xmax, ymax)
                        xmin = (xc - w / 2.0) * width
                        ymin = (yc - h / 2.0) * height
                        xmax = (xc + w / 2.0) * width
                        ymax = (yc + h / 2.0) * height

                        # Ensure valid bounding box dimensions
                        xmin = max(0, min(xmin, width - 1))
                        ymin = max(0, min(ymin, height - 1))
                        xmax = max(xmin + 1, min(xmax, width))
                        ymax = max(ymin + 1, min(ymax, height))

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        if len(boxes) == 0:
            # Handle empty annotation case
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([height, width])
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class COCOImageDataset(Dataset):
    """
    PyTorch Dataset for reading COCO format JSON annotations.
    """
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Map image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            self.img_to_anns.setdefault(img_id, []).append(ann)
            
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            # COCO bbox: [xmin, ymin, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    """
    Custom collate function for object detection batching.
    """
    return tuple(zip(*batch))
