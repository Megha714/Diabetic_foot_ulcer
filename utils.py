"""
Utility functions for DFU Detection project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device='auto'):
    """
    Get the best available device
    
    Args:
        device: 'auto', 'mps', 'cuda', or 'cpu'
    
    Returns:
        torch.device
    """
    if device == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true, y_pred, classes):
    """Print classification report"""
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4
    )
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    print("="*80 + "\n")


def save_training_metrics(metrics, save_path):
    """Save training metrics to file"""
    import json
    
    # Convert numpy types to Python types
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        elif isinstance(value, (list, tuple)):
            metrics_json[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
        else:
            metrics_json[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"💾 Saved metrics to {save_path}")


def visualize_predictions(model, dataloader, device, num_samples=16, save_path=None):
    """
    Visualize model predictions
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    confidences_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            output = model.predict(images)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu())
            preds_list.extend(output['predicted_class'].cpu())
            confidences_list.extend(output['confidence'].cpu())
            
            if len(images_list) >= num_samples:
                break
    
    # Plot
    num_samples = min(num_samples, len(images_list))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    class_names = ['Normal', 'Abnormal']
    
    for idx in range(num_samples):
        img = images_list[idx].permute(1, 2, 0).numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        
        true_label = class_names[labels_list[idx]]
        pred_label = class_names[preds_list[idx]]
        confidence = confidences_list[idx]
        
        color = 'green' if labels_list[idx] == preds_list[idx] else 'red'
        
        axes[idx].set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
            color=color,
            fontsize=10
        )
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved prediction visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_directory_structure(base_dir='./'):
    """Create project directory structure"""
    directories = [
        'checkpoints',
        'logs',
        'results',
        'visualizations',
        'data'
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
    
    print("✅ Created directory structure:")
    for directory in directories:
        print(f"  📁 {directory}/")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    return checkpoint


def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    
    # Inverse frequency weighting
    total_samples = len(labels)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)


def print_system_info():
    """Print system information"""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Device
    if torch.backends.mps.is_available():
        print(f"Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        print(f"Device: NVIDIA GPU (CUDA {torch.version.cuda})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"Device: CPU")
    
    # Python
    import sys
    print(f"Python version: {sys.version.split()[0]}")
    
    # NumPy
    print(f"NumPy version: {np.__version__}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test utilities
    print_system_info()
    
    # Test device selection
    device = get_device()
    print(f"Selected device: {device}")
    
    # Create directory structure
    create_directory_structure()
