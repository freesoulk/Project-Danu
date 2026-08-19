# PyTorch Custom Object Detection & Auto-Annotation Pipeline

An end-to-end Computer Vision & Machine Learning framework built on **PyTorch** and **Faster R-CNN** for custom object detection, automated image annotation/labelling, dataset formatting (YOLO & COCO), and IoU model evaluation.

---

## 🔬 Model Overview

This repository provides a modular, flexible object detection model builder:

- **Model Family**: **Faster R-CNN** (Faster Region-based Convolutional Neural Network).
- **Backbone Options**:
  - **MobileNetV3-Large FPN**: Ultra-lightweight backbone optimized for real-time CPU / edge inference.
  - **ResNet-50 FPN V2**: High-capacity backbone optimized for deeper feature representation and higher detection precision.
- **Architectural Features**:
  - **Feature Pyramid Network (FPN)**: Extracts multi-scale feature maps for accurate small-object (e.g. colony dots) and large-object detection.
  - **Region Proposal Network (RPN)**: Generates high-recall object proposals.
  - **RoI Heads & FastRCNNPredictor**: Performs bounding box regression and multi-class classification.
- **Domain Applications**:
  - Automated dataset annotation and pseudo-labeling.
  - Microbiological colony plate counting and morphological categorization (`small_dot` vs. `sparkling`).
  - Custom bounding box detection with automated YOLO `.txt` / COCO `.json` export.

---

## 📁 User-Friendly Directory Structure

```
.
├── input_images/                           # 📥 PLACE YOUR RAW IMAGES TO LABEL HERE
│   ├── sample_colony_dish.jpg
│   └── colony_plate_02.jpeg
├── annotated_outputs/                      # 📤 ANNOTATED IMAGES & YOLO LABELS GENERATED POST MODEL RUN
│   ├── annotated_images/                   # Visual preview images with bounding box overlays
│   │   ├── sample_colony_dish_annotated.png
│   │   └── colony_plate_02_annotated.png
│   ├── labels/                             # Exported YOLO .txt annotation files
│   ├── labels_small_dots/                  # Class-specific labels (small_dot)
│   ├── labels_sparklings/                  # Class-specific labels (sparkling)
│   ├── previews_small_dots/                # Class-specific visual preview renders
│   └── previews_sparklings/
├── training_dataset/                       # 🏋️ RESERVED FOR FUTURE MODEL TRAINING & VALIDATION DATA
│   ├── images/                             # train/ & val/ raw image folders
│   └── labels/                             # train/ & val/ ground-truth annotation folders
├── assets/                                 # Visual previews, sample images & training plots
├── scripts/                                # Utility & dataset preparation scripts
│   ├── annotate_colony_plate.py            # Computer Vision pre-annotator & rule-based extractor
│   └── create_sample_data.py               # Synthetic dataset generator
├── src/                                    # Core PyTorch source code
│   ├── auto_labeler.py                     # Auto-annotation engine & YOLO txt exporter
│   ├── dataset.py                          # PyTorch Dataset classes for YOLO & COCO formats
│   ├── evaluate.py                         # IoU evaluation engine & visual comparator
│   ├── model.py                            # Faster R-CNN detection model builder
│   └── train.py                            # PyTorch model training loop & loss visualizer
├── best_model.pth                          # Trained PyTorch model weights
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Image Labelling (Default Run)
Place raw images in `input_images/` and run:
```bash
python src/auto_labeler.py --input-dir input_images --output-dir annotated_outputs --conf-thresh 0.3
```
*Outputs:*
- High-res annotated images saved to `annotated_outputs/annotated_images/`
- YOLO `.txt` annotation files saved to `annotated_outputs/labels/`
- Class-separated outputs (`labels_small_dots/`, `labels_sparklings/`, `previews_small_dots/`, `previews_sparklings/`)

### 3. Pre-Annotate Colony Dish Images for Training
Extract annotations into `training_dataset/`:
```bash
python scripts/annotate_colony_plate.py --image-path assets/sample_colony_plate.jpg
```

### 4. Train Custom Detection Model (Future Training)
Train the PyTorch Faster R-CNN model when you add training/validation images to `training_dataset/`:
```bash
python src/train.py --data-dir training_dataset --epochs 5 --batch-size 4 --backbone mobilenet
```

### 5. Evaluate Model Performance
```bash
python src/evaluate.py --data-dir training_dataset --model-path best_model.pth --conf-thresh 0.3
```

---

## 🏷️ Supported Annotation Formats

- **YOLO Text Format**: `[class_id x_center y_center width height]` (normalized `0.0` to `1.0`)
- **COCO JSON Format**: `[xmin, ymin, width, height]`
- **Pascal VOC XML**: `<bndbox>` coordinates `[xmin, ymin, xmax, ymax]`

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
