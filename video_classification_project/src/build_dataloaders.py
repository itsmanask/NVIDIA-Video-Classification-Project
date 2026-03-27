"""
build_dataloaders.py

Defines VideoDataset and get_dataloaders().
Loads pre-normalized .pt tensor files from disk and returns
train, val, and test DataLoaders ready for model training.

Folder structure expected:
    frames_normalized/
        {split}/
            {class}/
                {subcategory}/
                    {video_name}.pt

Each .pt file contains a tensor of shape (16, 3, 224, 224)
    16  = number of frames
    3   = RGB channels
    224 = height
    224 = width

Labels are assigned by class folder name (not subcategory):
    Animation       → 0
    Flat_Content    → 1
    Gaming          → 2
    Natural_Content → 3

Usage from other scripts:
    from src.build_dataloaders import get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
    dataloaders = get_dataloaders(NORMALIZED_DIR, batch_size=32, num_workers=4)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


# ── FIX 4: Training augmentation transform ────────────────────────────
# Operates directly on normalized tensors — shape (16, 3, 224, 224)
# Applied ONLY to the training split (val/test get no augmentation)
def train_transform(tensor):
    """
    Lightweight augmentation for training tensors.

    Args:
        tensor : FloatTensor of shape (16, 3, 224, 224)

    Returns:
        tensor : augmented FloatTensor, same shape

    Augmentations:
        1. Random horizontal flip (prob=0.5)
           Flips all 16 frames consistently across width (dim=-1)
        2. Small Gaussian noise (std=0.01)
           Adds subtle noise to prevent overfitting without
           distorting the normalized pixel distribution meaningfully
    """
    # FIX 4a: Random horizontal flip across width dimension
    if torch.rand(1).item() < 0.5:
        tensor = tensor.flip(-1).contiguous()   # flip width dim: (16,3,224,224)

    # FIX 4b: Small Gaussian noise — magnitude kept very low (0.01)
    # so normalized range [-2.12, 2.64] is not meaningfully disturbed
    tensor = tensor + torch.randn_like(tensor) * 0.01

    return tensor


# ── Label Map ─────────────────────────────────────────────────────────
# Fixed forever — never change after training starts.
# Output neuron 0 always means Animation, etc.
CLASS_TO_IDX = {
    'Animation'      : 0,
    'Flat_Content'   : 1,
    'Gaming'         : 2,
    'Natural_Content': 3,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ── Dataset ───────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Loads pre-normalized video tensors from disk.

    Args:
        root_dir  : path to frames_normalized/{split} folder
        split     : 'train', 'val', or 'test' (used only for display)
        transform : optional transform to apply to each tensor

    Returns per sample:
        tensor : FloatTensor of shape (16, 3, 224, 224)
        label  : integer (0-3)
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform

        # List of (path_to_pt_file, label_integer) tuples
        self.samples = []

        self._build_sample_list()

    def _build_sample_list(self):
        """
        Walks the directory tree and collects all .pt file paths
        with their corresponding integer labels.

        Structure: root_dir / class / subcategory / video.pt
        Label comes from the class folder name.
        """
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"  WARNING: Missing class folder: {class_dir}")
                continue

            # Walk one level deeper — subcategory folders
            subcategories = sorted([
                d for d in os.listdir(class_dir)
                if os.path.isdir(os.path.join(class_dir, d))
            ])

            for subcategory in subcategories:
                sub_dir = os.path.join(class_dir, subcategory)

                pt_files = sorted([
                    f for f in os.listdir(sub_dir)
                    if f.endswith('.pt')
                ])

                for pt_file in pt_files:
                    full_path = os.path.join(sub_dir, pt_file)
                    self.samples.append((full_path, label))

        print(f"  [{self.split}] Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns one (tensor, label) pair.

        tensor shape : (16, 3, 224, 224)
        label        : integer 0-3
        """
        pt_path, label = self.samples[idx]

        tensor = torch.load(pt_path).contiguous()

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


# ── DataLoader Factory ────────────────────────────────────────────────
def get_dataloaders(base_dir, batch_size=32, num_workers=4):
    """
    Creates train, val, and test DataLoaders.

    Args:
        base_dir    : path to frames_normalized/
        batch_size  : number of videos per batch (32 is a good default)
        num_workers : parallel workers for loading (4 is safe on most machines)

    Returns:
        dict with keys 'train', 'val', 'test'
        each value is a DataLoader
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)

        # FIX 4: Apply augmentation only for training — val/test must be deterministic
        transform = train_transform if split == 'train' else None

        dataset = VideoDataset(
            root_dir  = split_dir,
            split     = split,
            transform = transform,
        )

        # Shuffle only for training — val and test must be deterministic
        shuffle = (split == 'train')

        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = False,   # faster GPU transfer
        )

        dataloaders[split] = loader

    return dataloaders


# ── Quick Sanity Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    BASE_DIR       = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
    NORMALIZED_DIR = os.path.join(BASE_DIR, 'frames_normalized')

    print("=" * 60)
    print("DATALOADER SANITY CHECK")
    print("=" * 60)

    dataloaders = get_dataloaders(NORMALIZED_DIR, batch_size=32, num_workers=4)

    for split, loader in dataloaders.items():
        # Grab one batch
        tensors, labels = next(iter(loader))

        print(f"\n[{split}]")
        print(f"  Batches in loader : {len(loader)}")
        print(f"  Batch tensor shape: {tensors.shape}")
        print(f"  Batch labels      : {labels.tolist()[:10]}...")
        print(f"  Label range       : min={labels.min()}  max={labels.max()}")
        print(f"  Tensor dtype      : {tensors.dtype}")
        print(f"  Tensor min/max    : {tensors.min():.3f} / {tensors.max():.3f}")

    print("\n" + "=" * 60)
    print("CLASS TO INDEX MAPPING")
    print("=" * 60)
    for cls, idx in CLASS_TO_IDX.items():
        print(f"  {idx}  →  {cls}")

    print("\nDataLoaders built and verified successfully.")
