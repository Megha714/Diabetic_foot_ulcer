"""
Enhanced Training Script with Proper Foot Validation
Trains model to distinguish feet from non-feet AND classify ulcers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

from model import HybridDFUModel
from enhanced_dataset import create_enhanced_data_loaders


class EnhancedDFUTrainer:
    """
    Enhanced trainer that properly trains foot validation
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.validation_criterion = nn.BCELoss()  # Binary classification for foot/non-foot
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_class_loss': [],
            'train_val_loss': [],
            'train_class_acc': [],
            'train_foot_acc': [],
            'val_loss': [],
            'val_class_acc': [],
            'val_foot_acc': [],
            'learning_rate': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Paths
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def compute_loss(self, outputs, batch):
        """
        Compute combined loss for classification and foot validation
        
        Args:
            outputs: Model outputs dict
            batch: Batch dict with 'class' and 'is_foot' labels
        """
        # Classification loss (only for foot images)
        foot_mask = batch['is_foot'] == 1
        
        if foot_mask.sum() > 0:
            foot_logits = outputs['classification_logits'][foot_mask]
            foot_labels = batch['class'][foot_mask]
            classification_loss = self.classification_criterion(foot_logits, foot_labels)
        else:
            classification_loss = torch.tensor(0.0, device=self.device)
        
        # Foot validation loss (all images)
        validation_loss = self.validation_criterion(
            outputs['validation_score'].squeeze(),
            batch['is_foot'].float()
        )
        
        # Combined loss
        total_loss = (
            self.config['classification_weight'] * classification_loss +
            self.config['validation_weight'] * validation_loss
        )
        
        return total_loss, classification_loss, validation_loss
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_val_loss = 0.0
        
        correct_class = 0
        total_class = 0
        correct_foot = 0
        total_foot = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['image'].to(self.device)
            class_labels = batch['class'].to(self.device)
            foot_labels = batch['is_foot'].to(self.device)
            
            batch_dict = {
                'class': class_labels,
                'is_foot': foot_labels
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, return_validation=True)
            
            # Compute loss
            loss, class_loss, val_loss = self.compute_loss(outputs, batch_dict)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_val_loss += val_loss.item()
            
            # Classification accuracy (only for foot images)
            foot_mask = foot_labels == 1
            if foot_mask.sum() > 0:
                foot_logits = outputs['classification_logits'][foot_mask]
                foot_class_labels = class_labels[foot_mask]
                _, predicted = torch.max(foot_logits, 1)
                correct_class += (predicted == foot_class_labels).sum().item()
                total_class += foot_mask.sum().item()
            
            # Foot validation accuracy (all images)
            foot_predictions = (outputs['validation_score'].squeeze() >= 0.5).long()
            correct_foot += (foot_predictions == foot_labels).sum().item()
            total_foot += len(foot_labels)
        
        # Averages
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_val_loss = total_val_loss / len(self.train_loader)
        class_acc = 100.0 * correct_class / total_class if total_class > 0 else 0.0
        foot_acc = 100.0 * correct_foot / total_foot
        
        return avg_loss, avg_class_loss, avg_val_loss, class_acc, foot_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct_class = 0
        total_class = 0
        correct_foot = 0
        total_foot = 0
        
        all_class_labels = []
        all_class_preds = []
        all_foot_labels = []
        all_foot_preds = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            class_labels = batch['class'].to(self.device)
            foot_labels = batch['is_foot'].to(self.device)
            
            batch_dict = {
                'class': class_labels,
                'is_foot': foot_labels
            }
            
            # Forward pass
            outputs = self.model(images, return_validation=True)
            
            # Compute loss
            loss, _, _ = self.compute_loss(outputs, batch_dict)
            total_loss += loss.item()
            
            # Classification metrics (only for foot images)
            foot_mask = foot_labels == 1
            if foot_mask.sum() > 0:
                foot_logits = outputs['classification_logits'][foot_mask]
                foot_class_labels = class_labels[foot_mask]
                _, predicted = torch.max(foot_logits, 1)
                
                correct_class += (predicted == foot_class_labels).sum().item()
                total_class += foot_mask.sum().item()
                
                all_class_labels.extend(foot_class_labels.cpu().numpy())
                all_class_preds.extend(predicted.cpu().numpy())
            
            # Foot validation metrics (all images)
            foot_predictions = (outputs['validation_score'].squeeze() >= 0.5).long()
            correct_foot += (foot_predictions == foot_labels).sum().item()
            total_foot += len(foot_labels)
            
            all_foot_labels.extend(foot_labels.cpu().numpy())
            all_foot_preds.extend(foot_predictions.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        class_acc = 100.0 * correct_class / total_class if total_class > 0 else 0.0
        foot_acc = 100.0 * correct_foot / total_foot
        
        # Detailed metrics
        metrics = {
            'loss': avg_loss,
            'class_accuracy': class_acc,
            'foot_accuracy': foot_acc
        }
        
        if len(all_class_labels) > 0:
            metrics['class_precision'] = precision_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
            metrics['class_recall'] = recall_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
            metrics['class_f1'] = f1_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
        
        if len(all_foot_labels) > 0:
            metrics['foot_precision'] = precision_score(all_foot_labels, all_foot_preds, zero_division=0)
            metrics['foot_recall'] = recall_score(all_foot_labels, all_foot_preds, zero_division=0)
            metrics['foot_f1'] = f1_score(all_foot_labels, all_foot_preds, zero_division=0)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"    💾 Saved new best model (Accuracy: {metrics['class_accuracy']:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("🚀 Starting Enhanced Training with Foot Validation")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_class_loss, train_val_loss, train_class_acc, train_foot_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_class_loss'].append(train_class_loss)
            self.history['train_val_loss'].append(train_val_loss)
            self.history['train_class_acc'].append(train_class_acc)
            self.history['train_foot_acc'].append(train_foot_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_class_acc'].append(val_metrics['class_accuracy'])
            self.history['val_foot_acc'].append(val_metrics['foot_accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Check if best
            is_best = val_metrics['class_accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['class_accuracy']
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch [{epoch+1}/{self.config['epochs']}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f} | Class Acc={train_class_acc:.2f}% | Foot Acc={train_foot_acc:.2f}%")
            print(f"  Val:   Loss={val_metrics['loss']:.4f} | Class Acc={val_metrics['class_accuracy']:.2f}% | Foot Acc={val_metrics['foot_accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if 'foot_f1' in val_metrics:
                print(f"  Foot Validation F1: {val_metrics['foot_f1']:.4f}")
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"✅ Training completed in {total_time/60:.1f} minutes")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print("="*80 + "\n")
        
        return self.history


def main():
    """Main training function"""
    
    # Configuration - optimized for 16GB RAM
    config = {
        'epochs': 30,
        'batch_size': 8,  # Reduced for stability
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        'classification_weight': 1.0,
        'validation_weight': 1.0,  # Equal weight to foot validation
        'checkpoint_dir': 'checkpoints',
        'num_workers': 0  # Disable multiprocessing to avoid hanging
    }
    
    print("\n" + "="*80)
    print("📊 Enhanced DFU Detection Training")
    print("   With Proper Foot Validation Training")
    print("="*80 + "\n")
    
    # Load data
    print("📁 Loading dataset...")
    train_loader, val_loader, test_loader, dataset_info = create_enhanced_data_loaders(
        positive_dir='DFU/Patches',
        negative_dir='DFU/Negative_Samples',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    if not dataset_info['has_negative_samples']:
        print("\n⚠️  WARNING: No negative samples found!")
        print("    The model will not learn to reject non-foot images properly.")
        print("    Please add negative samples to DFU/Negative_Samples/")
        print("\n    Do you want to continue training without negative samples?")
        response = input("    (y/n): ").strip().lower()
        if response != 'y':
            print("\n❌ Training cancelled. Please add negative samples first.")
            return
        print("\n⚠️  Continuing without negative samples (rejection will be weak)...\n")
    
    # Create model
    print("\n🔧 Creating model...")
    model = HybridDFUModel(
        num_classes=2,
        pretrained=True,
        rejection_threshold=0.70
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Create trainer
    trainer = EnhancedDFUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    print(f"   Device: {trainer.device}")
    
    # Train
    history = trainer.train()
    
    # Save training history
    history_path = Path(config['checkpoint_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Training history saved to {history_path}")
    print("\n✅ Training complete! You can now use the web app to test the model.")
    print("   The model should now properly reject non-foot images!")


if __name__ == '__main__':
    main()
