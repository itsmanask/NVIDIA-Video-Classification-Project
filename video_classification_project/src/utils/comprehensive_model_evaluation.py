import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_recall_fscore_support,
    roc_curve,
    auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from collections import defaultdict
import pandas as pd
from itertools import cycle
import sys

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to video_classification_project
data_dir = project_root / 'src' / 'data'

# Add to Python path
if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

# Import from your training script
try:
    from model_train_new import (
        SuperEnhancedTemporalModel as EnhancedCNNLSTM,
        EnhancedPreExtractedFeaturesDataset as MemoryEfficientVideoDataset,
        EnhancedTemporalModelTrainer as CheckpointManager,
        collate_features  # ✅ ADDED: Required for variable-length sequences
    )
    print("✓ Successfully imported from model_train_new.py")
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error importing from model_train_new.py: {e}")
    print(f"Looking in: {data_dir}")
    sys.exit(1)


class ComprehensiveEvaluator:
    """Combined model evaluation and data visualization"""
    
    def __init__(self, checkpoint_path, data_dir, output_dir=None, device='cuda', split='val'):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.data_dir = Path(data_dir)
        
        # Default output directory structure
        if output_dir is None:
            self.output_dir = Path('video_classification_project/results')
        else:
            self.output_dir = Path(output_dir)
        
        # Create subdirectories
        self.eval_dir = self.output_dir / 'evaluation'
        self.viz_dir = self.output_dir / 'visualizations'
        self.pred_dir = self.output_dir / 'predictions'
        self.data_viz_dir = self.output_dir / 'data_samples'
        
        for directory in [self.eval_dir, self.viz_dir, self.pred_dir, self.data_viz_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.split = split
        self.model = None
        self.category_mapping = None
        self.class_names = None
        self.num_classes = None
        
        # ImageNet mean and std for denormalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"{'='*80}")
        print(f"COMPREHENSIVE MODEL EVALUATION & VISUALIZATION")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Data Directory: {data_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Split: {split}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
    
    def denormalize_frame(self, tensor):
        """Convert normalized tensor back to displayable image"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        img = np.clip(img, 0, 1)
        return img
    
    def load_model_and_data(self):
        """Load trained model and dataset"""
        print(f"{'='*60}")
        print(f"LOADING MODEL AND DATA")
        print(f"{'='*60}\n")
        
        # Load checkpoint
        if self.checkpoint_path and self.checkpoint_path.exists():
            print(f"Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        else:
            print("No checkpoint provided - skipping model loading")
            checkpoint = None
        
        # Create dataset - this uses pre-extracted features in HDF5 format
        print(f"Loading {self.split} dataset...")
        
        # ✅ FIXED: Look for HDF5 feature files (not split directories)
        feature_file = self.data_dir / f'{self.split}_features.h5'
        if not feature_file.exists():
            # Try multiscale version
            feature_file = self.data_dir / f'{self.split}_features_multiscale.h5'
        
        if not feature_file.exists():
            raise ValueError(
                f"Feature file not found!\n"
                f"Expected: {self.data_dir / f'{self.split}_features.h5'}\n"
                f"Or: {self.data_dir / f'{self.split}_features_multiscale.h5'}\n"
                f"Please run feature extraction first (Stage 1 of training)."
            )
        
        # ✅ FIXED: Use correct dataset class with HDF5 file
        dataset = MemoryEfficientVideoDataset(
            feature_file=feature_file,
            augment=False
        )
        
        if len(dataset) == 0:
            raise ValueError(f"{self.split} dataset is empty!")
        
        # Get category mapping and feature dimension from dataset
        self.category_mapping = dataset.category_mapping
        self.num_classes = len(self.category_mapping)
        self.feature_dim = dataset.feature_dim  # ✅ ADDED: Get from dataset
        self.class_names = [None] * self.num_classes
        
        for name, idx in self.category_mapping.items():
            self.class_names[idx] = name
        
        print(f"\nCategories ({self.num_classes}):")
        for idx, name in enumerate(self.class_names):
            print(f"   {idx}: {name}")
        
        # Initialize model if checkpoint exists
        if checkpoint:
            print(f"\nInitializing model...")
            
            # ✅ FIXED: Get model configuration from checkpoint
            # Training saves as 'model_config', fallback to 'config' for compatibility
            config = checkpoint.get('model_config', checkpoint.get('config', {}))
            
            # ✅ FIXED: Use feature_dim from config or dataset (they should match)
            input_dim = config.get('feature_dim', self.feature_dim)
            hidden_dim = config.get('hidden_dim', 768)
            num_layers = config.get('num_lstm_layers', 4)
            num_attention_heads = config.get('num_attention_heads', 12)
            dropout = config.get('dropout', 0.4)
            bidirectional = config.get('bidirectional', True)
            
            print(f"   Model config from checkpoint:")
            print(f"      Feature dim: {input_dim}")
            print(f"      Hidden dim: {hidden_dim}")
            print(f"      LSTM layers: {num_layers}")
            print(f"      Attention heads: {num_attention_heads}")
            print(f"      Bidirectional: {bidirectional}")
            
            # ✅ FIXED: Create model with all correct parameters
            self.model = EnhancedCNNLSTM(
                feature_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_lstm_layers=num_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                bidirectional=bidirectional
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   Total parameters: {total_params:,}")
            print(f"   Epoch trained: {checkpoint.get('epoch', 'unknown')}")
            
            # Display validation accuracy from checkpoint
            if 'best_val_acc' in checkpoint:
                print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
            elif 'metrics' in checkpoint and 'accuracy' in checkpoint['metrics']:
                print(f"   Val Acc: {checkpoint['metrics']['accuracy']:.2f}%")
        
        # ✅ FIXED: Create data loader with custom collate function
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_features  # CRITICAL: Handles variable-length sequences
        )
        
        print(f"\n{self.split.capitalize()} dataset: {len(dataset):,} samples ({len(loader)} batches)\n")
        
        return dataset, loader
    
    def get_predictions(self, loader):
        """Get all predictions and ground truth labels"""
        print(f"{'='*60}")
        print(f"GENERATING PREDICTIONS")
        print(f"{'='*60}\n")
        
        if self.model is None:
            print("Model not loaded. Skipping predictions.")
            return None, None, None
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        
        with torch.no_grad():
            with tqdm(total=len(loader), desc="Predicting") as pbar:
                for batch in loader:
                    # ✅ FIXED: Unpack batch from collate_features
                    # Returns: (features, labels, lengths)
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        features, labels, lengths = batch
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                        features, labels = batch
                        lengths = None
                    else:
                        print(f"Warning: Unexpected batch format: {type(batch)}")
                        continue
                    
                    features = features.to(self.device)
                    if labels is not None:
                        labels = labels.to(self.device)
                    if lengths is not None:
                        lengths = lengths.to(self.device)
                    
                    # ✅ FIXED: Forward pass with lengths parameter
                    # Model signature: forward(self, x, lengths=None)
                    outputs = self.model(features, lengths)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    if labels is not None:
                        all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    pbar.update(1)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels) if all_labels else None
        all_probs = np.array(all_probs)
        
        print(f"Total predictions: {len(all_preds):,}\n")
        
        return all_preds, all_labels, all_probs
    
    # ===========================
    # EVALUATION VISUALIZATIONS
    # ===========================
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
            filename = 'confusion_matrix_normalized.png'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            filename = 'confusion_matrix.png'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.eval_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_per_category_metrics(self, y_true, y_pred):
        """Plot detailed per-category metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Category': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision
        axes[0, 0].barh(metrics_df['Category'], metrics_df['Precision'], color='skyblue')
        axes[0, 0].set_xlabel('Precision', fontweight='bold')
        axes[0, 0].set_title('Precision per Category', fontweight='bold')
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].axvline(x=metrics_df['Precision'].mean(), color='red', 
                           linestyle='--', label=f"Mean: {metrics_df['Precision'].mean():.3f}")
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Recall
        axes[0, 1].barh(metrics_df['Category'], metrics_df['Recall'], color='lightcoral')
        axes[0, 1].set_xlabel('Recall', fontweight='bold')
        axes[0, 1].set_title('Recall per Category', fontweight='bold')
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].axvline(x=metrics_df['Recall'].mean(), color='red', 
                          linestyle='--', label=f"Mean: {metrics_df['Recall'].mean():.3f}")
        axes[0, 1].legend()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # F1-Score
        axes[1, 0].barh(metrics_df['Category'], metrics_df['F1-Score'], color='lightgreen')
        axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
        axes[1, 0].set_title('F1-Score per Category', fontweight='bold')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].axvline(x=metrics_df['F1-Score'].mean(), color='red', 
                          linestyle='--', label=f"Mean: {metrics_df['F1-Score'].mean():.3f}")
        axes[1, 0].legend()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Support
        axes[1, 1].barh(metrics_df['Category'], metrics_df['Support'], color='plum')
        axes[1, 1].set_xlabel('Number of Samples', fontweight='bold')
        axes[1, 1].set_title('Support per Category', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for ax in axes.flat[:3]:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt='%d', padding=3)
        
        plt.suptitle('Per-Category Performance Metrics', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'per_category_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: per_category_metrics.png")
        plt.close()
        
        return metrics_df
    
    def plot_roc_curves(self, y_true, y_probs):
        """Plot ROC curves for each category"""
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(12, 10))
        
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
        
        for i, color in zip(range(self.num_classes), colors):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: roc_curves.png")
        plt.close()
        
        return roc_auc
    
    def plot_prediction_distribution(self, y_true, y_pred):
        """Plot prediction distribution and error analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        pred_counts = np.bincount(y_pred, minlength=self.num_classes)
        
        x = np.arange(self.num_classes)
        width = 0.35
        
        axes[0].bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
        axes[0].bar(x + width/2, pred_counts, width, label='Predicted', color='coral', alpha=0.8)
        axes[0].set_xlabel('Category', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        axes[0].set_title('True vs Predicted Distribution', fontweight='bold', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        errors = (y_true != y_pred).astype(int)
        error_by_class = np.array([errors[y_true == i].sum() for i in range(self.num_classes)])
        total_by_class = np.array([(y_true == i).sum() for i in range(self.num_classes)])
        error_rate = error_by_class / (total_by_class + 1e-10) * 100
        
        bars = axes[1].bar(self.class_names, error_rate, color='crimson', alpha=0.7)
        axes[1].set_xlabel('Category', fontweight='bold')
        axes[1].set_ylabel('Error Rate (%)', fontweight='bold')
        axes[1].set_title('Error Rate per Category', fontweight='bold', fontsize=14)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.eval_dir / 'prediction_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: prediction_distribution.png")
        plt.close()
    
    def plot_confidence_analysis(self, y_true, y_pred, y_probs):
        """Analyze prediction confidence"""
        confidence = np.max(y_probs, axis=1)
        correct = (y_true == y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confidence distribution
        axes[0, 0].hist(confidence[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0, 0].hist(confidence[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence', fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontweight='bold')
        axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Calibration
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(bins)-1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
            else:
                bin_accuracies.append(0)
        
        axes[0, 1].bar(bin_centers, bin_accuracies, width=0.08, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')
        axes[0, 1].set_xlabel('Confidence Bin', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Calibration: Accuracy vs Confidence', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Confidence per category
        category_confidence = []
        for i in range(self.num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                category_confidence.append(confidence[mask].mean())
            else:
                category_confidence.append(0)
        
        bars = axes[1, 0].bar(self.class_names, category_confidence, color='teal', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Category', fontweight='bold')
        axes[1, 0].set_ylabel('Average Confidence', fontweight='bold')
        axes[1, 0].set_title('Average Confidence per Category', fontweight='bold')
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Decision margin
        top2_probs = np.sort(y_probs, axis=1)[:, -2:]
        confidence_gap = top2_probs[:, 1] - top2_probs[:, 0]
        
        axes[1, 1].hist(confidence_gap[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[1, 1].hist(confidence_gap[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Confidence Gap (Top-1 - Top-2)', fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontweight='bold')
        axes[1, 1].set_title('Decision Margin: Correct vs Incorrect', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.eval_dir / 'confidence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: confidence_analysis.png")
        plt.close()
    
    # ===========================
    # DATA VISUALIZATIONS
    # ===========================
    
    def create_dataset_overview(self):
        """Create comprehensive dataset overview"""
        print(f"\n{'='*60}")
        print(f"CREATING DATASET OVERVIEW")
        print(f"{'='*60}\n")
        
        stats = {'train': {}, 'val': {}, 'test': {}}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            
            print(f"Scanning {split} split...")
            
            # Look for .pt feature files
            feature_files = list(split_dir.glob("**/*.pt"))
            
            for feature_file in feature_files:
                try:
                    # Extract category from path or filename
                    # Assuming structure: data_dir/split/category/file.pt
                    parts = feature_file.relative_to(split_dir).parts
                    if len(parts) >= 2:
                        category_name = parts[0]
                    else:
                        category_name = "unknown"
                    
                    if category_name not in stats[split]:
                        stats[split][category_name] = 0
                    
                    # Count as 1 sample per feature file
                    stats[split][category_name] += 1
                    
                except Exception as e:
                    continue
        
        # If no data found, skip visualization
        if not any(stats.values()):
            print("No feature data found. Skipping dataset overview.")
            print("Note: This visualization expects pre-extracted features in .pt files")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Category distribution per split
        ax = axes[0, 0]
        all_categories = set()
        for split_stats in stats.values():
            all_categories.update(split_stats.keys())
        all_categories = sorted(list(all_categories))
        
        x = np.arange(len(all_categories))
        width = 0.25
        
        for i, split in enumerate(['train', 'val', 'test']):
            counts = [stats[split].get(cat, 0) for cat in all_categories]
            ax.bar(x + i*width, counts, width, label=split.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Category', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Samples per Category by Split', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Total samples per split
        ax = axes[0, 1]
        split_totals = {split: sum(stats[split].values()) for split in ['train', 'val', 'test']}
        split_totals = {k: v for k, v in split_totals.items() if v > 0}  # Remove empty splits
        
        if split_totals:
            colors = ['#3b82f6', '#8b5cf6', '#ec4899'][:len(split_totals)]
            wedges, texts, autotexts = ax.pie(
                split_totals.values(), 
                labels=[f'{k.capitalize()}\n{v}' for k, v in split_totals.items()],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Dataset Split Distribution', fontweight='bold', fontsize=14)
        
        # Category proportions
        ax = axes[1, 0]
        category_totals = {}
        for split_stats in stats.values():
            for cat, count in split_stats.items():
                category_totals[cat] = category_totals.get(cat, 0) + count
        
        if category_totals:
            colors_cat = ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24']
            ax.barh(list(category_totals.keys()), list(category_totals.values()), 
                   color=colors_cat, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Total Samples', fontweight='bold')
            ax.set_title('Total Samples per Category', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
        
        # Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        total_samples = sum(split_totals.values())
        
        stats_text = [
            ["Metric", "Value"],
            ["─" * 30, "─" * 15],
            ["Total Samples", f"{total_samples:,}"],
            ["Categories", f"{len(all_categories)}"],
            ["Training Samples", f"{split_totals.get('train', 0):,}"],
            ["Validation Samples", f"{split_totals.get('val', 0):,}"],
            ["Test Samples", f"{split_totals.get('test', 0):,}"],
            ["Data Type", "Pre-extracted Features"],
            ["Feature Dim", "2048 (ResNet50)"]
        ]
        
        table = ax.table(cellText=stats_text, loc='center', cellLoc='left',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#374151')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(2, len(stats_text)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f3f4f6')
        
        ax.set_title('Dataset Statistics', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'dataset_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: dataset_overview.png\n")
        
        # Print text summary
        print(f"{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        for row in stats_text[2:]:
            print(f"  {row[0]:<25} {row[1]:>15}")
        
        # Category distribution per split
        ax = axes[0, 0]
        all_categories = set()
        for split_stats in stats.values():
            all_categories.update(split_stats.keys())
        all_categories = sorted(list(all_categories))
        
        x = np.arange(len(all_categories))
        width = 0.25
        
        for i, split in enumerate(['train', 'val', 'test']):
            counts = [stats[split].get(cat, 0) for cat in all_categories]
            ax.bar(x + i*width, counts, width, label=split.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Category', fontweight='bold')
        ax.set_ylabel('Number of Videos', fontweight='bold')
        ax.set_title('Videos per Category by Split', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Total videos per split
        ax = axes[0, 1]
        split_totals = {split: sum(stats[split].values()) for split in ['train', 'val', 'test']}
        colors = ['#3b82f6', '#8b5cf6', '#ec4899']
        wedges, texts, autotexts = ax.pie(
            split_totals.values(), 
            labels=[f'{k.capitalize()}\n{v}' for k, v in split_totals.items()],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('Dataset Split Distribution', fontweight='bold', fontsize=14)
        
        # Category proportions
        ax = axes[1, 0]
        category_totals = {}
        for split_stats in stats.values():
            for cat, count in split_stats.items():
                category_totals[cat] = category_totals.get(cat, 0) + count
        
        if category_totals:
            colors_cat = ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24']
            ax.barh(list(category_totals.keys()), list(category_totals.values()), 
                   color=colors_cat, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Total Videos', fontweight='bold')
            ax.set_title('Total Videos per Category', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
        
        # Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        total_videos = sum(split_totals.values())
        total_frames = total_videos * 32
        
        stats_text = [
            ["Metric", "Value"],
            ["─" * 30, "─" * 15],
            ["Total Videos", f"{total_videos:,}"],
            ["Total Frames", f"{total_frames:,}"],
            ["Categories", f"{len(all_categories)}"],
            ["Training Videos", f"{split_totals.get('train', 0):,}"],
            ["Validation Videos", f"{split_totals.get('val', 0):,}"],
            ["Test Videos", f"{split_totals.get('test', 0):,}"],
            ["Frames per Video", "32"],
            ["Frame Size", "224 × 224"],
            ["Normalization", "ImageNet"]
        ]
        
        table = ax.table(cellText=stats_text, loc='center', cellLoc='left',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#374151')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(2, len(stats_text)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f3f4f6')
        
        ax.set_title('Dataset Statistics', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'dataset_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: dataset_overview.png\n")
        
        # Print text summary
        print(f"{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        for row in stats_text[2:]:
            print(f"  {row[0]:<25} {row[1]:>15}")
        print()
    
    def visualize_sample_videos(self, split='train', samples_per_category=2):
        """Visualize sample videos from each category"""
        print(f"{'='*60}")
        print(f"NOTE: Video frame visualization skipped")
        print(f"      (Model uses pre-extracted features, not raw frames)")
        print(f"{'='*60}\n")
        return
        
        # Original visualization code commented out since we use features
        # This would require access to original video files
    
    def visualize_predictions_on_samples(self, dataset, num_samples=10):
        """Visualize model predictions on sample videos"""
        if self.model is None:
            print("Model not loaded. Skipping prediction visualization.")
            return
        
        print(f"{'='*60}")
        print(f"NOTE: Visual prediction display skipped")
        print(f"      (Model uses pre-extracted features)")
        print(f"      Predictions are shown in evaluation metrics.")
        print(f"{'='*60}\n")
        return
    
    def plot_prediction_with_frames(self, video, true_label, pred_label, probs, filename):
        """Plot video frames with prediction probabilities"""
        # Skipped - model uses features not frames
        pass
    
    def create_frame_montage(self, video_tensor, cols=8):
        """Create a montage of video frames"""
        # Skipped - model uses features not frames
        pass
    
    # ===========================
    # REPORTS
    # ===========================
    
    def generate_detailed_report(self, y_true, y_pred, y_probs, metrics_df, roc_auc):
        """Generate comprehensive text report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Checkpoint: {self.checkpoint_path}")
        report.append("")
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision = metrics_df['Precision'].mean()
        macro_recall = metrics_df['Recall'].mean()
        macro_f1 = metrics_df['F1-Score'].mean()
        
        report.append("OVERALL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Overall Accuracy: {accuracy*100:.2f}%")
        report.append(f"Macro Precision:  {macro_precision:.4f}")
        report.append(f"Macro Recall:     {macro_recall:.4f}")
        report.append(f"Macro F1-Score:   {macro_f1:.4f}")
        report.append("")
        
        # Per-category performance
        report.append("PER-CATEGORY PERFORMANCE")
        report.append("-"*80)
        report.append(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        report.append("-"*80)
        
        for _, row in metrics_df.iterrows():
            report.append(
                f"{row['Category']:<30} "
                f"{row['Precision']:>10.4f} "
                f"{row['Recall']:>10.4f} "
                f"{row['F1-Score']:>10.4f} "
                f"{int(row['Support']):>10d}"
            )
        report.append("")
        
        # ROC AUC scores
        report.append("ROC AUC SCORES")
        report.append("-"*80)
        for i, class_name in enumerate(self.class_names):
            report.append(f"{class_name:<30} {roc_auc[i]:.4f}")
        report.append(f"{'Macro Average':<30} {np.mean(list(roc_auc.values())):.4f}")
        report.append("")
        
        # Analysis
        report.append("ANALYSIS")
        report.append("-"*80)
        best_f1_idx = metrics_df['F1-Score'].idxmax()
        worst_f1_idx = metrics_df['F1-Score'].idxmin()
        
        report.append(f"Best performing category:  {metrics_df.loc[best_f1_idx, 'Category']} "
                     f"(F1: {metrics_df.loc[best_f1_idx, 'F1-Score']:.4f})")
        report.append(f"Worst performing category: {metrics_df.loc[worst_f1_idx, 'Category']} "
                     f"(F1: {metrics_df.loc[worst_f1_idx, 'F1-Score']:.4f})")
        
        # Confidence statistics
        confidence = np.max(y_probs, axis=1)
        correct = (y_true == y_pred)
        
        report.append("")
        report.append(f"Average confidence (correct):   {confidence[correct].mean():.4f}")
        report.append(f"Average confidence (incorrect): {confidence[~correct].mean():.4f}")
        report.append("")
        
        # Confusion matrix insights
        cm = confusion_matrix(y_true, y_pred)
        report.append("COMMON MISCLASSIFICATIONS (Top 5)")
        report.append("-"*80)
        
        misclass = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    misclass.append((cm[i, j], self.class_names[i], self.class_names[j]))
        
        misclass.sort(reverse=True)
        for count, true_class, pred_class in misclass[:5]:
            report.append(f"{true_class} -> {pred_class}: {count} samples")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Saved: evaluation_report.txt\n")
        print(report_text)
        
        return report_text
    
    def save_results_json(self, y_true, y_pred, y_probs, metrics_df, roc_auc):
        """Save results in JSON format"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(self.checkpoint_path),
            'overall_metrics': {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'macro_precision': float(metrics_df['Precision'].mean()),
                'macro_recall': float(metrics_df['Recall'].mean()),
                'macro_f1': float(metrics_df['F1-Score'].mean())
            },
            'per_category_metrics': {
                row['Category']: {
                    'precision': float(row['Precision']),
                    'recall': float(row['Recall']),
                    'f1_score': float(row['F1-Score']),
                    'support': int(row['Support']),
                    'roc_auc': float(roc_auc[idx])
                }
                for idx, row in metrics_df.iterrows()
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'class_names': self.class_names
        }
        
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved: evaluation_results.json")
    
    # ===========================
    # MAIN EXECUTION
    # ===========================
    
    def run_full_evaluation(self):
        """Run complete evaluation and visualization pipeline"""
        print(f"{'='*80}")
        print(f"STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*80}\n")
        
        # 1. Load model and data
        dataset, loader = self.load_model_and_data()
        
        # 2. Create dataset overview
        self.create_dataset_overview()
        
        # 3. Note: Video frame visualization skipped (using features not frames)
        print(f"{'='*60}")
        print(f"DATA VISUALIZATION")
        print(f"{'='*60}")
        print("Note: Raw video frame visualization not available")
        print("      (Model uses pre-extracted features)")
        print()
        
        # 4. Get predictions (if model loaded)
        if self.model is not None:
            y_pred, y_true, y_probs = self.get_predictions(loader)
            
            if y_pred is not None and y_true is not None:
                # 5. Generate evaluation visualizations
                print(f"{'='*60}")
                print(f"GENERATING EVALUATION METRICS")
                print(f"{'='*60}\n")
                
                self.plot_confusion_matrix(y_true, y_pred, normalize=False)
                self.plot_confusion_matrix(y_true, y_pred, normalize=True)
                
                metrics_df = self.plot_per_category_metrics(y_true, y_pred)
                roc_auc = self.plot_roc_curves(y_true, y_probs)
                self.plot_prediction_distribution(y_true, y_pred)
                self.plot_confidence_analysis(y_true, y_pred, y_probs)
                
                print()
                
                # 6. Generate reports
                print(f"{'='*60}")
                print(f"GENERATING REPORTS")
                print(f"{'='*60}\n")
                
                self.generate_detailed_report(y_true, y_pred, y_probs, metrics_df, roc_auc)
                self.save_results_json(y_true, y_pred, y_probs, metrics_df, roc_auc)
            else:
                print("⚠ Could not generate predictions - check data format")
        else:
            print("⚠ No model loaded - only dataset overview generated")
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"\nDirectory structure:")
        print(f"   {self.output_dir}/")
        if self.model is not None:
            print(f"   ├── evaluation/          (performance metrics)")
        print(f"   ├── visualizations/      (dataset overview)")
        if self.model is not None:
            print(f"   ├── evaluation_report.txt")
            print(f"   └── evaluation_results.json")
        print()
        
        # Print generated files
        if self.model is not None:
            print("Generated files:")
            eval_files = list(self.eval_dir.glob('*.png'))
            if eval_files:
                print(f"\nEvaluation Metrics ({len(eval_files)} files):")
                for f in sorted(eval_files):
                    print(f"   • {f.name}")
        
        viz_files = list(self.viz_dir.glob('*.png'))
        if viz_files:
            print(f"\nVisualizations ({len(viz_files)} files):")
            for f in sorted(viz_files):
                print(f"   • {f.name}")
        
        if self.model is not None:
            report_file = self.output_dir / 'evaluation_report.txt'
            json_file = self.output_dir / 'evaluation_results.json'
            if report_file.exists():
                print(f"\nReports:")
                print(f"   • evaluation_report.txt")
            if json_file.exists():
                print(f"   • evaluation_results.json")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation and visualization')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional for data-only viz)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='video_classification_project/data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='video_classification_project/results',
        help='Directory to save all results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("\nSearching for available checkpoints...")
            
            possible_dirs = [
                Path('video_classification_project/models/checkpoints'),
                Path('models/checkpoints'),
                Path('checkpoints')
            ]
            
            for checkpoint_dir in possible_dirs:
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob('*.pt'))
                    if checkpoints:
                        print(f"\nIn {checkpoint_dir}:")
                        for ckpt in sorted(checkpoints)[:10]:
                            print(f"   {ckpt}")
            return
    
    # Create evaluator and run
    evaluator = ComprehensiveEvaluator(
        checkpoint_path=args.checkpoint,
        data_dir=data_dir,
        output_dir=args.output_dir,
        device=args.device,
        split=args.split
    )
    
    try:
        evaluator.run_full_evaluation()
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("="*80)
        print("COMPREHENSIVE EVALUATION - AUTO MODE")
        print("="*80)
        print("\nSearching for checkpoint and data...")
        
        # Auto-detect checkpoint
        possible_checkpoints = [
            Path('video_classification_project/models/checkpoints/best_model.pt'),
            Path('models/checkpoints/best_model.pt'),
            Path('checkpoints/best_model.pt'),
            Path('best_model.pt')
        ]
        
        checkpoint_path = None
        for path in possible_checkpoints:
            if path.exists():
                checkpoint_path = path
                print(f"✓ Found checkpoint: {path}")
                break
        
        if checkpoint_path is None:
            print("⚠ No checkpoint found - will run data visualization only")
        
        # Auto-detect data directory
        data_dir_candidates = [
            Path('video_classification_project/data/processed'),
            Path('data/processed'),
            Path('../data/processed'),
            Path('processed')
        ]
        
        data_dir = None
        for path in data_dir_candidates:
            if path.exists():
                data_dir = path
                print(f"✓ Found data directory: {path}")
                break
        
        if data_dir is None:
            print("\n❌ Data directory not found.")
            print("Please specify with --data_dir argument")
            sys.exit(1)
        
        # Run evaluation
        print(f"\n{'='*80}")
        print("STARTING AUTO EVALUATION")
        print(f"{'='*80}\n")
        
        evaluator = ComprehensiveEvaluator(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            output_dir='video_classification_project/results',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            split='val'
        )
        
        try:
            evaluator.run_full_evaluation()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        main()