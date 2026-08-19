import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torchvision
# pyrefly: ignore [missing-import]
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detection_model(
    num_classes: int,
    pretrained: bool = True,
    backbone_type: str = "mobilenet",
    min_size: int = 400,
    max_size: int = 800
) -> nn.Module:
    """
    Constructs a PyTorch Faster R-CNN object detection model for custom image labelling.
    
    Args:
        num_classes (int): Total classes including background (class 0 is background).
        pretrained (bool): Whether to use pretrained weights.
        backbone_type (str): 'mobilenet' for fast inference or 'resnet50' for higher accuracy.
        min_size (int): Minimum image side size for model resizer.
        max_size (int): Maximum image side size for model resizer.
        
    Returns:
        torch.nn.Module: Configured Faster R-CNN model ready for training or inference.
    """
    if backbone_type == "mobilenet":
        try:
            weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                min_size=min_size,
                max_size=max_size
            )
        except (AttributeError, TypeError):
            # Fallback for older torchvision versions
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=pretrained,
                min_size=min_size,
                max_size=max_size
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif backbone_type == "resnet50":
        try:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                min_size=min_size,
                max_size=max_size
            )
        except (AttributeError, TypeError):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                min_size=min_size,
                max_size=max_size
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone_type: '{backbone_type}'. Choose 'mobilenet' or 'resnet50'.")

    return model

def get_classification_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constructs a ResNet-50 multi-label or multi-class image classification model.
    Outputs raw logits suitable for CrossEntropyLoss or BCEWithLogitsLoss.
    """
    try:
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
    except (AttributeError, TypeError):
        model = torchvision.models.resnet50(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    # Quick test initialization
    test_model = get_detection_model(num_classes=5, pretrained=False, backbone_type="mobilenet")
    num_params = sum(p.numel() for p in test_model.parameters())
    print(f"Model initialized successfully. Total parameters: {num_params:,}")

