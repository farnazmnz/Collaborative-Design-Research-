"""
Model Comparison Script
Compares UNet and Attention UNet predictions and saves individual output images
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from data_preprocessing import create_dataloaders
from unet import UNet
from attention_unet import AttentionUNet


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()


def iou_score(pred, target, threshold=0.5):
    """Calculate IoU score"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou.item()


def save_predictions(data_dir='brain_tumor_dataset/data',
                     unet_checkpoint='checkpoints/unet/unet_best.pth',
                     attention_unet_checkpoint='checkpoints/attention_unet/attention_unet_best.pth',
                     output_dir='predictions',
                     num_samples=20,
                     batch_size=4):
    """
    Save predicted segmentation outputs as individual image files

    Args:
        data_dir: Path to data directory
        unet_checkpoint: Path to UNet checkpoint
        attention_unet_checkpoint: Path to Attention UNet checkpoint
        output_dir: Directory to save predictions
        num_samples: Number of samples to process
        batch_size: Batch size for processing
    """

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    unet_output_dir = os.path.join(output_dir, 'unet')
    attention_unet_output_dir = os.path.join(output_dir, 'attention_unet')
    comparison_dir = os.path.join(output_dir, 'comparison')

    os.makedirs(unet_output_dir, exist_ok=True)
    os.makedirs(attention_unet_output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    unet_model = UNet(in_channels=1, out_channels=1, init_features=64).to(device)
    attention_unet_model = AttentionUNet(in_channels=1, out_channels=1, init_features=64).to(device)

    unet_ckpt = torch.load(unet_checkpoint, map_location=device)
    attention_unet_ckpt = torch.load(attention_unet_checkpoint, map_location=device)

    unet_model.load_state_dict(unet_ckpt['model_state_dict'])
    attention_unet_model.load_state_dict(attention_unet_ckpt['model_state_dict'])

    unet_model.eval()
    attention_unet_model.eval()

    print(f"UNet - Best Dice: {unet_ckpt['best_dice']:.4f}")
    print(f"Attention UNet - Best Dice: {attention_unet_ckpt['best_dice']:.4f}")

    # Load test data
    print("\nLoading test data...")
    dataloaders = create_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    test_loader = dataloaders['test']

    # Process samples
    print(f"\nProcessing {num_samples} samples...")
    print("="*80)

    sample_count = 0
    all_metrics = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            if sample_count >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            unet_outputs = torch.sigmoid(unet_model(images))
            attention_unet_outputs = torch.sigmoid(attention_unet_model(images))

            # Process each sample in batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break

                img = images[i, 0].cpu().numpy()
                true_mask = masks[i, 0].cpu().numpy()
                unet_pred = unet_outputs[i, 0].cpu().numpy()
                attention_unet_pred = attention_unet_outputs[i, 0].cpu().numpy()

                # Threshold predictions
                unet_pred_binary = (unet_pred > 0.5).astype(np.uint8) * 255
                attention_unet_pred_binary = (attention_unet_pred > 0.5).astype(np.uint8) * 255
                true_mask_binary = (true_mask > 0.5).astype(np.uint8) * 255

                # Normalize input image for saving
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

                # Calculate metrics
                unet_dice = dice_coefficient(
                    torch.from_numpy(unet_pred),
                    torch.from_numpy(true_mask)
                )
                attention_dice = dice_coefficient(
                    torch.from_numpy(attention_unet_pred),
                    torch.from_numpy(true_mask)
                )

                unet_iou = iou_score(
                    torch.from_numpy(unet_pred),
                    torch.from_numpy(true_mask)
                )
                attention_iou = iou_score(
                    torch.from_numpy(attention_unet_pred),
                    torch.from_numpy(true_mask)
                )

                # Save metrics
                all_metrics.append({
                    'sample': sample_count,
                    'unet_dice': unet_dice,
                    'unet_iou': unet_iou,
                    'attention_dice': attention_dice,
                    'attention_iou': attention_iou
                })

                # Save individual images
                sample_name = f'sample_{sample_count:03d}'

                # Save input image
                Image.fromarray(img_norm).save(
                    os.path.join(output_dir, f'{sample_name}_input.png')
                )

                # Save ground truth
                Image.fromarray(true_mask_binary).save(
                    os.path.join(output_dir, f'{sample_name}_ground_truth.png')
                )

                # Save UNet prediction
                Image.fromarray(unet_pred_binary).save(
                    os.path.join(unet_output_dir, f'{sample_name}_unet.png')
                )

                # Save Attention UNet prediction
                Image.fromarray(attention_unet_pred_binary).save(
                    os.path.join(attention_unet_output_dir, f'{sample_name}_attention_unet.png')
                )

                # Create and save comparison figure
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))

                axes[0].imshow(img_norm, cmap='gray')
                axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                axes[1].imshow(true_mask_binary, cmap='gray')
                axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[1].axis('off')

                axes[2].imshow(unet_pred_binary, cmap='gray')
                axes[2].set_title(f'UNet\nDice: {unet_dice:.3f} | IoU: {unet_iou:.3f}',
                                 fontsize=11, fontweight='bold')
                axes[2].axis('off')

                axes[3].imshow(attention_unet_pred_binary, cmap='gray')
                axes[3].set_title(f'Attention UNet\nDice: {attention_dice:.3f} | IoU: {attention_iou:.3f}',
                                 fontsize=11, fontweight='bold')
                axes[3].axis('off')

                # Overlay comparison
                overlay = np.stack([img_norm/255., img_norm/255., img_norm/255.], axis=-1)
                overlay[unet_pred_binary > 127] = [0, 1, 0]  # Green for UNet
                overlay[attention_unet_pred_binary > 127] = [1, 0, 0]  # Red for Attention UNet
                overlay[true_mask_binary > 127] = [0, 0, 1]  # Blue for ground truth

                axes[4].imshow(overlay)
                axes[4].set_title('Overlay\n(Blue=GT, Green=UNet, Red=AttUNet)',
                                 fontsize=10, fontweight='bold')
                axes[4].axis('off')

                plt.suptitle(f'Sample {sample_count}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(comparison_dir, f'{sample_name}_comparison.png'),
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()

                # Print progress
                print(f"Sample {sample_count:3d} - "
                      f"UNet: Dice={unet_dice:.3f}, IoU={unet_iou:.3f} | "
                      f"AttUNet: Dice={attention_dice:.3f}, IoU={attention_iou:.3f}")

                sample_count += 1

    # Calculate overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    unet_dice_avg = np.mean([m['unet_dice'] for m in all_metrics])
    unet_iou_avg = np.mean([m['unet_iou'] for m in all_metrics])
    attention_dice_avg = np.mean([m['attention_dice'] for m in all_metrics])
    attention_iou_avg = np.mean([m['attention_iou'] for m in all_metrics])

    print(f"\nUNet:")
    print(f"  Average Dice: {unet_dice_avg:.4f}")
    print(f"  Average IoU: {unet_iou_avg:.4f}")

    print(f"\nAttention UNet:")
    print(f"  Average Dice: {attention_dice_avg:.4f}")
    print(f"  Average IoU: {attention_iou_avg:.4f}")

    print(f"\nDifference:")
    print(f"  Dice: {(attention_dice_avg - unet_dice_avg):.4f}")
    print(f"  IoU: {(attention_iou_avg - unet_iou_avg):.4f}")

    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("SAMPLE-WISE METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Sample':<10} {'UNet Dice':<12} {'UNet IoU':<12} {'AttUNet Dice':<14} {'AttUNet IoU':<14}\n")
        f.write("-"*80 + "\n")
        for m in all_metrics:
            f.write(f"{m['sample']:<10} {m['unet_dice']:<12.4f} {m['unet_iou']:<12.4f} "
                   f"{m['attention_dice']:<14.4f} {m['attention_iou']:<14.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"\nUNet:\n")
        f.write(f"  Average Dice: {unet_dice_avg:.4f}\n")
        f.write(f"  Average IoU: {unet_iou_avg:.4f}\n")
        f.write(f"\nAttention UNet:\n")
        f.write(f"  Average Dice: {attention_dice_avg:.4f}\n")
        f.write(f"  Average IoU: {attention_iou_avg:.4f}\n")
        f.write(f"\nDifference:\n")
        f.write(f"  Dice: {(attention_dice_avg - unet_dice_avg):.4f}\n")
        f.write(f"  IoU: {(attention_iou_avg - unet_iou_avg):.4f}\n")

    print(f"\nMetrics saved to: {metrics_file}")
    print("\n" + "="*80)
    print("FILES SAVED")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    print(f"  - {sample_count} input images: sample_XXX_input.png")
    print(f"  - {sample_count} ground truth masks: sample_XXX_ground_truth.png")
    print(f"  - {sample_count} UNet predictions: unet/sample_XXX_unet.png")
    print(f"  - {sample_count} Attention UNet predictions: attention_unet/sample_XXX_attention_unet.png")
    print(f"  - {sample_count} comparison images: comparison/sample_XXX_comparison.png")
    print(f"  - Metrics file: metrics.txt")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare UNet and Attention UNet predictions')
    parser.add_argument('--data_dir', type=str, default='brain_tumor_dataset/data',
                        help='Path to data directory')
    parser.add_argument('--unet_checkpoint', type=str, default='checkpoints/unet/unet_best.pth',
                        help='Path to UNet checkpoint')
    parser.add_argument('--attention_unet_checkpoint', type=str,
                        default='checkpoints/attention_unet/attention_unet_best.pth',
                        help='Path to Attention UNet checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')

    args = parser.parse_args()

    save_predictions(
        data_dir=args.data_dir,
        unet_checkpoint=args.unet_checkpoint,
        attention_unet_checkpoint=args.attention_unet_checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
