import sys
from pathlib import Path
import os
import time
import argparse
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
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

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(img.convert("RGB") if hasattr(img, "convert") else img for img in images)
        
        # Convert PIL images to Tensors if not already
        image_tensors = []
        for img in images:
            if not isinstance(img, torch.Tensor):
                # Convert PIL Image to Tensor [3, H, W], normalized 0..1
                img_tensor = torch.from_numpy(
                    __import__("numpy").array(img)
                ).permute(2, 0, 1).float() / 255.0
                image_tensors.append(img_tensor.to(device))
            else:
                image_tensors.append(img.to(device))

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(image_tensors, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
            print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{num_batches}] - Loss: {losses.item():.4f}")

    avg_loss = total_loss / num_batches
    elapsed = time.time() - start_time
    print(f"--> Epoch [{epoch+1}] Completed in {elapsed:.2f}s | Average Train Loss: {avg_loss:.4f}\n")
    return avg_loss

@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    Computes validation loss by enabling training mode temporarily for loss dictionary calculation.
    """
    model.train() # Faster R-CNN requires train mode to calculate loss_dict
    total_loss = 0.0
    num_batches = len(data_loader)

    for images, targets in data_loader:
        image_tensors = []
        for img in images:
            if not isinstance(img, torch.Tensor):
                img_tensor = torch.from_numpy(
                    __import__("numpy").array(img)
                ).permute(2, 0, 1).float() / 255.0
                image_tensors.append(img_tensor.to(device))
            else:
                image_tensors.append(img.to(device))

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(image_tensors, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    avg_val_loss = total_loss / max(1, num_batches)
    return avg_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train custom Image Labelling / Object Detection model.")
    parser.add_argument("--data-dir", type=str, default="training_dataset", help="Root data directory")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes including background (0=bg)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="mobilenet", choices=["mobilenet", "resnet50"], help="Backbone model")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Dataset & Dataloaders
    train_img_dir = os.path.join(args.data_dir, "images", "train")
    train_lbl_dir = os.path.join(args.data_dir, "labels", "train")
    val_img_dir = os.path.join(args.data_dir, "images", "val")
    val_lbl_dir = os.path.join(args.data_dir, "labels", "val")
    if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
        try:
            from scripts.create_sample_data import generate_sample_dataset
        except ImportError:
            from create_sample_data import generate_sample_dataset
        generate_sample_dataset(output_dir=args.data_dir)

    train_dataset = YOLOImageDataset(train_img_dir, train_lbl_dir)
    val_dataset = YOLOImageDataset(val_img_dir, val_lbl_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset Loaded: {len(train_dataset)} train images, {len(val_dataset)} val images.")

    # Model, Optimizer, Scheduler
    model = get_detection_model(num_classes=args.num_classes, pretrained=True, backbone_type=args.backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\n--- Starting Training for {args.epochs} Epochs ---")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": args.num_classes,
                "backbone": args.backbone,
                "val_loss": val_loss
            }
            torch.save(checkpoint, args.save_path)
            print(f"--> Saved best model checkpoint to '{args.save_path}' (Val Loss: {val_loss:.4f})")

    # Plot training curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Image Labelling Model Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = "training_loss_plot.png"
    plt.savefig(plot_path)
    print(f"\nSaved loss curves plot to '{plot_path}'")
    print("--- Training Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
