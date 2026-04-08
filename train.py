import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data_preprocessing import create_dataloaders
from unet import UNet
from attention_unet import AttentionUNet


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""

    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


def dice_coefficient(predictions, targets, threshold=0.5):
    """Calculate Dice coefficient for evaluation"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)

    return dice.item()


def iou_score(predictions, targets, threshold=0.5):
    """Calculate IoU (Intersection over Union) score"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    iou = intersection / (union + 1e-8)

    return iou.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks)
        total_iou += iou_score(outputs, masks)

        pbar.set_postfix({
            'loss': loss.item(),
            'dice': dice_coefficient(outputs, masks),
            'iou': iou_score(outputs, masks)
        })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)

            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice_coefficient(outputs, masks),
                'iou': iou_score(outputs, masks)
            })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def test_model(model, dataloader, criterion, device):
    """Test the model on test dataset"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)

            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice_coefficient(outputs, masks),
                'iou': iou_score(outputs, masks)
            })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def save_checkpoint(model, optimizer, epoch, best_dice, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_dice = checkpoint['best_dice']
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {epoch}, Best Dice: {best_dice:.4f}")
    return epoch, best_dice


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Dice coefficient plot
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True)

    # IoU plot
    axes[2].plot(history['train_iou'], label='Train IoU')
    axes[2].plot(history['val_iou'], label='Val IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].set_title('Training and Validation IoU')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def visualize_predictions(model, dataloader, device, save_path, num_samples=4):
    """Visualize model predictions"""
    model.eval()

    images, masks = next(iter(dataloader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        img = images[i, 0].cpu().numpy()
        true_mask = masks[i, 0].cpu().numpy()
        pred_mask = predictions[i, 0].cpu().numpy()

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


def train(model_type='unet', epochs=50, batch_size=8, learning_rate=1e-4,
          data_dir='brain_tumor_dataset/data', save_dir='checkpoints'):
    """
    Main training function

    Args:
        model_type: 'unet' or 'attention_unet'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        data_dir: Path to data directory
        save_dir: Path to save checkpoints and results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'unet':
        model = UNet(in_channels=1, out_channels=1, init_features=64)
    elif model_type == 'attention_unet':
        model = AttentionUNet(in_channels=1, out_channels=1, init_features=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': []
    }

    best_dice = 0.0

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )

        # Validate
        val_loss, val_dice, val_iou = validate_epoch(
            model, dataloaders['val'], criterion, device
        )

        # Update learning rate
        scheduler.step(val_dice)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                os.path.join(model_save_dir, f'{model_type}_best.pth')
            )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                os.path.join(model_save_dir, f'{model_type}_epoch_{epoch + 1}.pth')
            )

    # Save final model
    save_checkpoint(
        model, optimizer, epochs - 1, best_dice,
        os.path.join(model_save_dir, f'{model_type}_final.pth')
    )

    # Plot training history
    plot_training_history(
        history,
        os.path.join(model_save_dir, f'{model_type}_training_history.png')
    )

    # Visualize predictions on validation set
    visualize_predictions(
        model, dataloaders['val'], device,
        os.path.join(model_save_dir, f'{model_type}_predictions.png')
    )

    # Test the model
    print("\n" + "="*50)
    print("Testing the model on test dataset...")
    print("="*50)
    test_loss, test_dice, test_iou = test_model(
        model, dataloaders['test'], criterion, device
    )
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU: {test_iou:.4f}")

    # Save test results
    with open(os.path.join(model_save_dir, f'{model_type}_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")

    print(f"\nTraining complete! Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to {model_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train brain tumor segmentation model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attention_unet'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='brain_tumor_dataset/data',
                        help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Path to save checkpoints')

    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
