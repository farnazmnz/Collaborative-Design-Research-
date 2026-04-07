import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, file_indices, transform=None, augment=False):
        """
        Brain Tumor Segmentation Dataset

        Args:
            data_dir: Path to the directory containing .mat files
            file_indices: List of file indices to include in this dataset
            transform: Optional transform to be applied on images
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.file_indices = file_indices
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        mat_file = os.path.join(self.data_dir, f'{file_idx}.mat')

        # Load the .mat file using h5py (for MATLAB v7.3 format)
        with h5py.File(mat_file, 'r') as f:
            image = np.array(f['cjdata']['image']).astype(np.float32)
            mask = np.array(f['cjdata']['tumorMask']).astype(np.float32)

        # Normalize image to [0, 1]
        image = self.normalize_image(image)

        # Binary mask (0 or 1)
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

    def normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        return image

    def apply_augmentation(self, image, mask):
        """Apply random augmentation to image and mask"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        return image, mask


def get_file_indices(data_dir):
    """Get all available file indices from the data directory"""
    files = os.listdir(data_dir)
    mat_files = [f for f in files if f.endswith('.mat')]
    indices = [int(f.replace('.mat', '')) for f in mat_files]
    indices.sort()
    return indices


def create_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create train, validation, and test splits

    Args:
        data_dir: Path to the directory containing .mat files
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing train, val, and test indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Get all file indices
    all_indices = get_file_indices(data_dir)

    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=random_seed
    )

    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_adjusted,
        random_state=random_seed
    )

    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


def create_dataloaders(data_dir, batch_size=8, num_workers=4, train_ratio=0.7,
                       val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Path to the directory containing .mat files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    # Create data splits
    splits = create_data_splits(data_dir, train_ratio, val_ratio, test_ratio, random_seed)

    # Create datasets
    train_dataset = BrainTumorDataset(
        data_dir,
        splits['train'],
        augment=True
    )

    val_dataset = BrainTumorDataset(
        data_dir,
        splits['val'],
        augment=False
    )

    test_dataset = BrainTumorDataset(
        data_dir,
        splits['test'],
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test the data loading
    data_dir = 'brain_tumor_dataset/data'

    print("Creating data splits...")
    splits = create_data_splits(data_dir)
    print(f"Train samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")

    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(data_dir, batch_size=4, num_workers=0)

    print("\nTesting data loading...")
    for split_name, loader in dataloaders.items():
        images, masks = next(iter(loader))
        print(f"{split_name.capitalize()} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask unique values: {torch.unique(masks)}")
