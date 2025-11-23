"""
Enhanced Dataset with Negative Samples for Foot Validation
Includes real foot images (positive) + random objects/body parts (negative)
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DFUDatasetWithValidation(Dataset):
    """
    Enhanced DFU Dataset that includes foot validation labels
    - Foot images (from your dataset): is_foot = 1, class = Normal/Abnormal
    - Non-foot images (to be added): is_foot = 0, class = ignored
    """
    def __init__(
        self, 
        root_dir, 
        is_foot_label=1,  # 1 for foot images, 0 for non-foot
        transform=None, 
        class_to_idx=None
    ):
        """
        Args:
            root_dir: Directory containing images
            is_foot_label: 1 if directory contains foot images, 0 otherwise
            transform: Albumentations transform
            class_to_idx: Mapping of class names to indices
        """
        self.root_dir = Path(root_dir)
        self.is_foot_label = is_foot_label
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all samples from directory"""
        if not self.root_dir.exists():
            print(f"⚠️ Warning: Directory {self.root_dir} does not exist")
            return
        
        # If it's a foot image directory, load with class labels
        if self.is_foot_label == 1:
            # Expect subdirectories: Normal, Abnormal
            for class_dir in self.root_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                # Map class names (check "abnormal" BEFORE "normal" since "abnormal" contains "normal")
                if 'abnormal' in class_name.lower() or 'ulcer' in class_name.lower():
                    class_idx = 1  # Abnormal
                elif 'normal' in class_name.lower() or 'healthy' in class_name.lower():
                    class_idx = 0  # Normal
                else:
                    continue
                
                # Load images from this class
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append({
                            'path': str(img_path),
                            'class': class_idx,
                            'is_foot': 1
                        })
        
        # If it's a non-foot directory, load without class labels
        else:
            for img_path in self.root_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append({
                        'path': str(img_path),
                        'class': -1,  # No class label for non-foot images
                        'is_foot': 0
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Basic transform if none provided
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = transform(image)
        
        return {
            'image': image,
            'class': sample['class'],
            'is_foot': sample['is_foot'],
            'path': sample['path']
        }


def get_enhanced_transforms(img_size=224, augment=True):
    """
    Get transforms for training with augmentation
    """
    if augment:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_negative_samples_instructions():
    """
    Print instructions for adding negative samples
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                   📋 ADDING NEGATIVE SAMPLES                             ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    To make your model reject non-foot images, you need to add negative samples.
    
    📁 Create this folder structure:
    
    DFU/
      ├── Patches/                    (Your existing dataset - KEEP AS IS)
      │     ├── Normal(Healthy skin)/
      │     └── Abnormal(Ulcer)/
      │
      └── Negative_Samples/           (NEW - Add non-foot images here)
            ├── objects/              (bottles, cars, furniture, food, etc.)
            ├── body_parts/           (hands, arms, legs, face, etc.)
            ├── animals/              (dogs, cats, birds, etc.)
            ├── nature/               (landscapes, trees, flowers, etc.)
            └── graphics/             (logos, text, diagrams, screenshots, etc.)
    
    🎯 What to include:
    
    1. **Common Objects** (50-100 images):
       - Bottles, glasses, cups
       - Cars, bicycles, vehicles
       - Furniture, electronics
       - Food items
       - Tools, equipment
    
    2. **Other Body Parts** (50-100 images):
       - Hands with/without wounds
       - Arms with/without wounds
       - Legs (not feet)
       - Face, torso
       - Any non-foot body part
    
    3. **Animals** (30-50 images):
       - Dog paws (important!)
       - Cat paws
       - Any animals
    
    4. **Nature & Scenes** (30-50 images):
       - Landscapes
       - Buildings
       - Nature close-ups
       - Abstract patterns
    
    5. **Graphics & Text** (30-50 images):
       - Screenshots
       - Logos, diagrams
       - Text-heavy images
       - Charts, graphs
    
    📊 Recommended: 200-300 negative samples total
    ✅ Minimum: 100 negative samples
    
    💡 Tips:
    - Use diverse, high-quality images
    - Include challenging cases (like dog paws)
    - More negative samples = better rejection
    - You can download from Google Images or use stock photo sites
    
    🔧 Quick Download Sources:
    - Google Images (search "random objects", "hands", "dog paws")
    - Unsplash.com (free high-quality photos)
    - Pexels.com (free stock photos)
    - Your own phone photos
    
    ⚠️ Important:
    - Do NOT modify your existing Patches folder
    - Only add new images to Negative_Samples
    - Images will be automatically resized to 224x224
    - Any image format works (.jpg, .png, .bmp)
    
    After adding negative samples, run:
        python3 train_with_validation.py
    
    This will retrain the model to properly reject non-foot images! 🚀
    """
    
    return instructions


def create_enhanced_data_loaders(
    positive_dir='DFU/Patches',
    negative_dir='DFU/Negative_Samples',
    batch_size=16,
    num_workers=4,
    val_split=0.15,
    test_split=0.15,
    seed=42
):
    """
    Create data loaders with both positive (foot) and negative (non-foot) samples
    Uses STRATIFIED splitting to ensure class balance in train/val/test sets.
    
    Args:
        positive_dir: Directory with foot images (Normal/Abnormal subfolders)
        negative_dir: Directory with non-foot images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    from sklearn.model_selection import train_test_split
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Transforms
    train_transform = get_enhanced_transforms(224, augment=True)
    val_transform = get_enhanced_transforms(224, augment=False)
    
    # Load positive samples (foot images with ulcer classification)
    positive_dataset = DFUDatasetWithValidation(
        root_dir=positive_dir,
        is_foot_label=1,
        transform=train_transform
    )
    
    print(f"✅ Loaded {len(positive_dataset)} foot images")
    
    # Check if negative samples exist
    negative_path = Path(negative_dir)
    if negative_path.exists() and any(negative_path.rglob('*.jpg')) or any(negative_path.rglob('*.png')):
        negative_dataset = DFUDatasetWithValidation(
            root_dir=negative_dir,
            is_foot_label=0,
            transform=train_transform
        )
        print(f"✅ Loaded {len(negative_dataset)} non-foot images")
        
        # Combine datasets
        combined_dataset = ConcatDataset([positive_dataset, negative_dataset])
        total_samples = len(combined_dataset)
        
        has_negative_samples = True
    else:
        print("⚠️ No negative samples found!")
        print("   Model will be trained only on foot images.")
        print("   Rejection mechanism will be weak without negative samples.")
        print("\n" + create_negative_samples_instructions())
        
        combined_dataset = positive_dataset
        total_samples = len(combined_dataset)
        has_negative_samples = False
    
    # STRATIFIED split - get labels efficiently without loading images
    print("📊 Preparing stratified split...")
    all_indices = []
    all_labels = []
    
    # Get labels from positive dataset (foot images)
    for idx in range(len(positive_dataset)):
        sample_info = positive_dataset.samples[idx]
        cls = sample_info['class']
        all_indices.append(idx)
        if cls == 0:
            all_labels.append('foot_normal')
        else:
            all_labels.append('foot_abnormal')
    
    # Get labels from negative dataset (non-foot images)
    if has_negative_samples:
        offset = len(positive_dataset)
        for idx in range(len(negative_dataset)):
            all_indices.append(offset + idx)
            all_labels.append('non_foot')
    
    print(f"   Classes: {len([l for l in all_labels if l=='foot_normal'])} normal, "
          f"{len([l for l in all_labels if l=='foot_abnormal'])} abnormal, "
          f"{len([l for l in all_labels if l=='non_foot'])} non-foot")
    
    # First split: train vs (val+test)
    train_indices, temp_indices, _, temp_labels = train_test_split(
        all_indices, all_labels,
        test_size=(val_split + test_split),
        stratify=all_labels,
        random_state=seed
    )
    
    # Second split: val vs test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_split/(val_split + test_split),
        stratify=temp_labels,
        random_state=seed
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(combined_dataset, test_indices)
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print(f"✅ Split complete: {train_size} train, {val_size} val, {test_size} test")
    
    # Create data loaders
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
    
    dataset_info = {
        'total_samples': total_samples,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'positive_samples': len(positive_dataset),
        'negative_samples': len(negative_dataset) if has_negative_samples else 0,
        'has_negative_samples': has_negative_samples,
        'batch_size': batch_size
    }
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Testing Enhanced Dataset with Negative Samples")
    print("="*80 + "\n")
    
    # Test loading
    train_loader, val_loader, test_loader, info = create_enhanced_data_loaders(
        batch_size=8,
        num_workers=0
    )
    
    print("\n📊 Dataset Information:")
    print(f"   Total samples: {info['total_samples']}")
    print(f"   - Positive (foot): {info['positive_samples']}")
    print(f"   - Negative (non-foot): {info['negative_samples']}")
    print(f"   Train: {info['train_samples']}")
    print(f"   Val: {info['val_samples']}")
    print(f"   Test: {info['test_samples']}")
    print(f"   Has negative samples: {'Yes ✅' if info['has_negative_samples'] else 'No ❌'}")
    
    # Test a batch
    print("\n🧪 Testing data loading...")
    batch = next(iter(train_loader))
    print(f"   Batch shape: {batch['image'].shape}")
    print(f"   Class labels: {batch['class']}")
    print(f"   Foot labels: {batch['is_foot']}")
    print("\n✅ Dataset ready!")
