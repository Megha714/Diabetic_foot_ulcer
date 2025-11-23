"""
Custom Vision Transformer for DFU Detection with Rejection Mechanism
Author: AI-Powered Medical Image Analysis
Description: Hybrid ViT architecture with foot validation and ulcer detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Tuple, Dict


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to create patches
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        attn_output, attn_weights = self.attn(self.norm1(x))
        x = x + attn_output
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class FootValidationModule(nn.Module):
    """Module to validate if image contains a foot"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.validation_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, embed_dim) - CLS token representation
        return self.validation_head(x)


class DFUViT(nn.Module):
    """
    Custom Vision Transformer for DFU Detection
    Features:
    - Foot validation (rejection mechanism)
    - DFU classification (Normal vs Abnormal)
    - Multi-head attention for interpretability
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        rejection_threshold=0.7
    ):
        super().__init__()
        
        self.rejection_threshold = rejection_threshold
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Foot validation head (rejection mechanism)
        self.foot_validator = FootValidationModule(embed_dim)
        
        # DFU classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Create patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        x = self.norm(x)
        
        return x, attention_weights
    
    def forward(self, x, return_validation=True):
        # Extract features
        features, attention_weights = self.forward_features(x)
        
        # Get CLS token representation
        cls_output = features[:, 0]
        
        # Foot validation score
        validation_score = self.foot_validator(cls_output)
        
        # DFU classification
        classification_logits = self.classification_head(cls_output)
        
        if return_validation:
            return {
                'classification_logits': classification_logits,
                'validation_score': validation_score,
                'attention_weights': attention_weights[-1],  # Last layer attention
                'features': cls_output
            }
        else:
            return classification_logits
    
    def predict(self, x):
        """Inference with rejection mechanism"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_validation=True)
            
            validation_score = output['validation_score']
            classification_logits = output['classification_logits']
            
            # Apply rejection threshold
            is_valid = validation_score >= self.rejection_threshold
            
            # Get predictions
            probs = F.softmax(classification_logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            return {
                'is_valid_foot': is_valid,
                'validation_confidence': validation_score,
                'predicted_class': predicted_class,
                'class_probabilities': probs,
                'confidence': confidence,
                'rejected': ~is_valid
            }


class HybridDFUModel(nn.Module):
    """
    Hybrid model combining pre-trained ViT with custom architecture
    Uses transfer learning for better performance
    """
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        rejection_threshold=0.7
    ):
        super().__init__()
        
        self.rejection_threshold = rejection_threshold
        
        # Load pre-trained ViT
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        embed_dim = self.backbone.embed_dim
        
        # Foot validation head
        self.foot_validator = FootValidationModule(embed_dim)
        
        # DFU classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_validation=True):
        # Extract features using pre-trained ViT
        features = self.backbone(x)  # (B, embed_dim)
        
        # Foot validation
        validation_score = self.foot_validator(features)
        
        # Classification
        classification_logits = self.classification_head(features)
        
        if return_validation:
            return {
                'classification_logits': classification_logits,
                'validation_score': validation_score,
                'features': features
            }
        else:
            return classification_logits
    
    def predict(self, x):
        """Inference with rejection mechanism"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_validation=True)
            
            validation_score = output['validation_score']
            classification_logits = output['classification_logits']
            
            # Apply rejection threshold
            is_valid = validation_score >= self.rejection_threshold
            
            # Get predictions
            probs = F.softmax(classification_logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            return {
                'is_valid_foot': is_valid,
                'validation_confidence': validation_score,
                'predicted_class': predicted_class,
                'class_probabilities': probs,
                'confidence': confidence,
                'rejected': ~is_valid
            }


def get_model(model_type='hybrid', **kwargs):
    """Factory function to create models"""
    if model_type == 'hybrid':
        return HybridDFUModel(**kwargs)
    elif model_type == 'custom':
        return DFUViT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Test hybrid model
    model = HybridDFUModel(num_classes=2, pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    
    output = model(x)
    print("\nHybrid Model Output:")
    print(f"Classification logits shape: {output['classification_logits'].shape}")
    print(f"Validation score shape: {output['validation_score'].shape}")
    
    # Test prediction
    pred = model.predict(x)
    print(f"\nPrediction:")
    print(f"Valid foot: {pred['is_valid_foot']}")
    print(f"Predicted class: {pred['predicted_class']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
