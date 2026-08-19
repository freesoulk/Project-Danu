# Project-Danu
PyTorch Faster R-CNN Auto-Labelling &amp; Object Detection Model  An end-to-end Computer Vision machine learning model designed for automated bounding box detection, custom image annotation, and dataset label generation (YOLO &amp; COCO formats).


🔑 Key Highlights
Architecture: Faster R-CNN (Faster Region-based Convolutional Neural Network) paired with a Feature Pyramid Network (FPN) for multi-scale object detection.
Dual Backbones:
⚡ MobileNetV3-Large FPN: Lightweight backbone optimized for high-speed CPU / edge inference.
🎯 ResNet-50 FPN V2: High-capacity backbone optimized for maximum detection accuracy.

Primary Capabilities:
Automated Labelling: Detects objects on unlabelled images and automatically exports normalized YOLO .txt files and visual preview renders.
Class-Specific Filtering: Categorizes and separates detections per class (e.g., distinguishing small_dot vs. sparkling micro-colonies).
Performance Metrics: Computes IoU (Intersection over Union) and Recall with side-by-side ground truth vs. prediction comparisons.
