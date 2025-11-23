import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
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
import glob

warnings.filterwarnings('ignore')

# Add safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


class MultiscaleH5Dataset(Dataset):
    """Dataset for multiscale features stored in H5 format"""
    
    def __init__(self, h5_path, split='val'):
        self.h5_path = Path(h5_path)
        self.split = split
        
        print(f"Loading features from: {self.h5_path}")
        
        # Open H5 file and inspect structure
        with h5py.File(self.h5_path, 'r') as f:
            print(f"\nH5 file structure:")
            print(f"  Keys: {list(f.keys())}")
            
            # Check if features is a dataset or group
            if 'features' in f:
                features_obj = f['features']
                print(f"  'features' type: {type(features_obj)}")
                
                if isinstance(features_obj, h5py.Dataset):
                    # Features is a single dataset (array)
                    print(f"  Features shape: {features_obj.shape}")
                    self.features = features_obj[:]
                    
                    # Get labels
                    if 'labels' in f:
                        self.labels = f['labels'][:]
                        print(f"  Labels shape: {self.labels.shape}")
                    elif 'categories' in f:
                        self.labels = f['categories'][:]
                        print(f"  Categories shape: {self.labels.shape}")
                    else:
                        raise ValueError("No labels found in H5 file")
                    
                    # Get video names/paths if available
                    if 'video_names' in f:
                        self.video_names = [name.decode() if isinstance(name, bytes) else name 
                                          for name in f['video_names'][:]]
                    elif 'video_paths' in f:
                        self.video_names = [name.decode() if isinstance(name, bytes) else name 
                                          for name in f['video_paths'][:]]
                    else:
                        self.video_names = [f"video_{i}" for i in range(len(self.features))]
                    
                    print(f"  Number of videos: {len(self.video_names)}")
                    
                else:
                    # Features is a group with nested structure
                    raise ValueError(
                        "Expected 'features' to be a dataset array, but got a group. "
                        "Please check your H5 file structure."
                    )
            else:
                raise ValueError(f"No 'features' key found in H5 file. Available keys: {list(f.keys())}")
            
            # Create category mapping from labels
            unique_labels = sorted(set(self.labels))
            self.num_classes = len(unique_labels)
            
            # If labels are already integers, use them directly
            if all(isinstance(label, (int, np.integer)) for label in unique_labels):
                self.category_mapping = {f"class_{i}": i for i in unique_labels}
            else:
                # Labels are strings/categories
                self.category_mapping = {str(cat): idx for idx, cat in enumerate(unique_labels)}
                # Convert labels to indices
                cat_to_idx = {cat: idx for idx, cat in enumerate(unique_labels)}
                self.labels = np.array([cat_to_idx[label] for label in self.labels])
            
            # Determine feature dimension
            if self.features.ndim == 3:
                # Shape: (num_videos, seq_len, feature_dim)
                self.feature_dim = self.features.shape[-1]
            elif self.features.ndim == 2:
                # Shape: (num_videos, feature_dim) - single frame features
                self.feature_dim = self.features.shape[-1]
                # Add sequence dimension
                self.features = self.features[:, np.newaxis, :]
            else:
                raise ValueError(f"Unexpected features shape: {self.features.shape}")
        
        print(f"\nCategory mapping:")
        for name, idx in sorted(self.category_mapping.items(), key=lambda x: x[1]):
            count = sum(1 for label in self.labels if label == idx)
            print(f"   {idx}: {name} ({count} samples)")
        
        print(f"\nDataset info:")
        print(f"  Total videos: {len(self.video_names)}")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        # Features are already loaded in memory
        features = torch.from_numpy(self.features[idx].astype(np.float32))
        label = int(self.labels[idx])
        
        return features, label


def collate_features(batch):
    """Collate function for variable-length sequences"""
    features_list, labels = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([f.shape[0] for f in features_list])
    
    # Pad sequences to max length in batch
    max_len = lengths.max().item()
    feature_dim = features_list[0].shape[-1]
    
    padded_features = torch.zeros(len(features_list), max_len, feature_dim)
    
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len] = features
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, labels, lengths


class SuperEnhancedTemporalModel(nn.Module):
    """Match the exact architecture from model_train_new.py"""
    
    def __init__(self, input_dim=1280, hidden_dim=768, num_classes=4, 
                 num_lstm_layers=4, dropout=0.4, num_attention_heads=12,
                 intermediate_dim=512, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_lstm_layers
        self.bidirectional = bidirectional
        
        # Input projection (not feature_proj)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Multi-head attention
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.MultiheadAttention(
            lstm_output_dim,
            num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(lstm_output_dim)
        
        # Attention-based pooling (match training code exactly)
        self.attention_pooling = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Enhanced classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.LayerNorm(intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, num_classes)
        )
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        device = x.device  # Store device before packing
        
        # Project input
        x = self.input_projection(x)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)  # Residual connection
        
        # Attention-based pooling
        attention_weights = self.attention_pooling(attn_out)
        
        if lengths is not None:
            # Create mask using the stored device
            mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
            attention_weights = attention_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        pooled = (attn_out * attention_weights).sum(1)
        
        # Classify
        output = self.classifier(pooled)
        
        return output


class ComprehensiveEvaluator:
    """Model evaluation with H5 features support"""
    
    def __init__(self, checkpoint_path, h5_path, output_dir, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.h5_path = Path(h5_path)
        self.output_dir = Path(output_dir)
        
        # Create subdirectories
        self.eval_dir = self.output_dir / 'evaluation'
        self.viz_dir = self.output_dir / 'visualizations'
        
        for directory in [self.eval_dir, self.viz_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.category_mapping = None
        self.class_names = None
        self.num_classes = None
        
        print(f"{'='*80}")
        print(f"COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"H5 Features: {h5_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
    
    def load_model_and_data(self):
        """Load trained model and dataset"""
        print(f"{'='*60}")
        print(f"LOADING MODEL AND DATA")
        print(f"{'='*60}\n")
        
        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            print("✓ Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
        
        # Create dataset
        print(f"\nLoading dataset from H5...")
        dataset = MultiscaleH5Dataset(self.h5_path)
        
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty!")
        
        # Get metadata
        self.category_mapping = dataset.category_mapping
        self.num_classes = len(self.category_mapping)
        self.feature_dim = dataset.feature_dim
        
        self.class_names = [None] * self.num_classes
        for name, idx in self.category_mapping.items():
            self.class_names[idx] = name
        
        # Initialize model
        print(f"\nInitializing model...")
        
        # Get config from checkpoint
        config = checkpoint.get('model_config', checkpoint.get('config', {}))
        
        input_dim = config.get('input_dim', config.get('feature_dim', self.feature_dim))
        hidden_dim = config.get('hidden_dim', 768)
        num_layers = config.get('num_lstm_layers', 4)
        num_attention_heads = config.get('num_attention_heads', 12)
        dropout = config.get('dropout', 0.4)
        bidirectional = config.get('bidirectional', True)
        intermediate_dim = config.get('intermediate_dim', 512)
        
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   LSTM layers: {num_layers}")
        print(f"   Attention heads: {num_attention_heads}")
        print(f"   Intermediate dim: {intermediate_dim}")
        
        # Create model with correct architecture
        self.model = SuperEnhancedTemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_lstm_layers=num_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            bidirectional=bidirectional,
            intermediate_dim=intermediate_dim
        ).to(self.device)
        
        # Load state dict
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✓ Model weights loaded successfully")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   Total parameters: {total_params:,}")
            
            if 'best_val_acc' in checkpoint:
                print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
            elif 'val_acc' in checkpoint:
                print(f"   Val Acc: {checkpoint['val_acc']:.2f}%")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_features
        )
        
        print(f"\nDataset: {len(dataset):,} samples\n")
        
        return dataset, loader
    
    def get_predictions(self, loader):
        """Get all predictions and ground truth labels"""
        print(f"{'='*60}")
        print(f"GENERATING PREDICTIONS")
        print(f"{'='*60}\n")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        
        with torch.no_grad():
            with tqdm(total=len(loader), desc="Predicting") as pbar:
                for batch in loader:
                    features, labels, lengths = batch
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(features, lengths)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    pbar.update(1)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print(f"Total predictions: {len(all_preds):,}\n")
        
        return all_preds, all_labels, all_probs
    
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
        
        save_path = self.viz_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
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
        }).sort_values('F1-Score', ascending=True)
        
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
        
        plt.suptitle('Per-Category Performance Metrics', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.viz_dir / 'per_category_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: per_category_metrics.png")
        plt.close()
        
        return metrics_df
    
    def plot_roc_curves(self, y_true, y_probs):
        """Plot ROC curves"""
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(12, 10))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        
        for i, color in zip(range(self.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.viz_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: roc_curves.png")
        plt.close()
        
        return roc_auc
    
    def generate_report(self, y_true, y_pred, y_probs, metrics_df, roc_auc):
        """Generate comprehensive report"""
        report = []
        report.append("="*80)
        report.append("MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Checkpoint: {self.checkpoint_path}")
        report.append(f"H5 File: {self.h5_path}")
        report.append("")
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        report.append("OVERALL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Accuracy: {accuracy*100:.2f}%")
        report.append(f"Macro Precision: {metrics_df['Precision'].mean():.4f}")
        report.append(f"Macro Recall: {metrics_df['Recall'].mean():.4f}")
        report.append(f"Macro F1-Score: {metrics_df['F1-Score'].mean():.4f}")
        report.append("")
        
        # Per-category
        report.append("PER-CATEGORY PERFORMANCE")
        report.append("-"*80)
        report.append(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        report.append("-"*80)
        
        for _, row in metrics_df.iterrows():
            report.append(
                f"{row['Category']:<20} "
                f"{row['Precision']:>10.4f} "
                f"{row['Recall']:>10.4f} "
                f"{row['F1-Score']:>10.4f} "
                f"{int(row['Support']):>10d}"
            )
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.eval_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Saved: evaluation_report.txt\n")
        print(report_text)
        
        return report_text
    
    def run_full_evaluation(self):
        """Run complete evaluation"""
        print(f"\n{'='*80}")
        print(f"STARTING EVALUATION")
        print(f"{'='*80}\n")
        
        # Load model and data
        dataset, loader = self.load_model_and_data()
        
        # Get predictions
        y_pred, y_true, y_probs = self.get_predictions(loader)
        
        # Generate visualizations
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")
        
        self.plot_confusion_matrix(y_true, y_pred, normalize=False)
        self.plot_confusion_matrix(y_true, y_pred, normalize=True)
        metrics_df = self.plot_per_category_metrics(y_true, y_pred)
        roc_auc = self.plot_roc_curves(y_true, y_probs)
        
        # Generate report
        print(f"\n{'='*60}")
        print(f"GENERATING REPORT")
        print(f"{'='*60}\n")
        
        self.generate_report(y_true, y_pred, y_probs, metrics_df, roc_auc)
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {self.output_dir}")
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': metrics_df['Precision'].mean(),
            'recall': metrics_df['Recall'].mean(),
            'f1': metrics_df['F1-Score'].mean()
        }


def find_all_ensemble_models(models_dir):
    """Find all ensemble model checkpoints"""
    models_path = Path(models_dir)
    
    # Look for patterns: best_ensemble_model_*.pt
    pattern = 'best_ensemble_model_*.pt'
    model_files = sorted(models_path.glob(pattern))
    
    if not model_files:
        print(f"⚠ No ensemble models found matching pattern: {pattern}")
        print(f"Looking in: {models_path}")
        return []
    
    # Extract model numbers and sort
    models = []
    for model_file in model_files:
        # Extract number from filename
        try:
            # best_ensemble_model_0.pt -> 0
            num_str = model_file.stem.split('_')[-1]
            model_num = int(num_str)
            models.append((model_num, model_file))
        except:
            continue
    
    # Sort by model number
    models.sort(key=lambda x: x[0])
    
    return models


def evaluate_all_models(models_dir, h5_file, base_output_dir, device='cuda'):
    """Evaluate all ensemble models sequentially"""
    
    print(f"\n{'#'*80}")
    print(f"# MULTI-MODEL EVALUATION SYSTEM")
    print(f"{'#'*80}\n")
    
    # Find all models
    print(f"Searching for ensemble models in: {models_dir}")
    models = find_all_ensemble_models(models_dir)
    
    if not models:
        print("❌ No models found to evaluate!")
        return
    
    print(f"\n✓ Found {len(models)} ensemble models:")
    for model_num, model_path in models:
        print(f"   Model {model_num}: {model_path.name}")
    
    print(f"\nH5 Features: {h5_file}")
    print(f"Base Output Directory: {base_output_dir}")
    print(f"Device: {device}")
    
    # Create summary results
    summary_results = []
    
    # Evaluate each model
    for model_num, model_path in models:
        print(f"\n\n{'='*80}")
        print(f"{'='*80}")
        print(f"EVALUATING MODEL {model_num}")
        print(f"{'='*80}")
        print(f"{'='*80}\n")
        
        # Create model-specific output directory
        model_output_dir = Path(base_output_dir) / f'model_{model_num}'
        
        try:
            # Create evaluator for this model
            evaluator = ComprehensiveEvaluator(
                checkpoint_path=model_path,
                h5_path=h5_file,
                output_dir=model_output_dir,
                device=device
            )
            
            # Run evaluation
            results = evaluator.run_full_evaluation()
            
            # Store results
            summary_results.append({
                'model_num': model_num,
                'model_path': str(model_path),
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'output_dir': str(model_output_dir)
            })
            
            print(f"\n✅ Model {model_num} evaluation completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error evaluating Model {model_num}: {e}")
            import traceback
            traceback.print_exc()
            
            summary_results.append({
                'model_num': model_num,
                'model_path': str(model_path),
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'error': str(e),
                'output_dir': str(model_output_dir)
            })
    
    # Generate summary report
    print(f"\n\n{'#'*80}")
    print(f"# EVALUATION SUMMARY")
    print(f"{'#'*80}\n")
    
    generate_summary_report(summary_results, base_output_dir)
    
    print(f"\n{'#'*80}")
    print(f"# ALL EVALUATIONS COMPLETE")
    print(f"{'#'*80}")
    print(f"\nTotal models evaluated: {len(models)}")
    print(f"Results directory: {base_output_dir}")


def generate_summary_report(summary_results, output_dir):
    """Generate summary report comparing all models"""
    
    output_dir = Path(output_dir)
    
    # Create summary dataframe
    df = pd.DataFrame(summary_results)
    
    # Sort by model number
    df = df.sort_values('model_num')
    
    # Print summary table
    print("="*100)
    print(f"{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Status':<10}")
    print("="*100)
    
    for _, row in df.iterrows():
        if pd.notna(row.get('accuracy')):
            print(f"{row['model_num']:<10} "
                  f"{row['accuracy']*100:>10.2f}%  "
                  f"{row['precision']:>10.4f}  "
                  f"{row['recall']:>10.4f}  "
                  f"{row['f1']:>10.4f}  "
                  f"{'✓ Success':<10}")
        else:
            error_msg = row.get('error', 'Unknown error')[:30]
            print(f"{row['model_num']:<10} "
                  f"{'N/A':>11}  "
                  f"{'N/A':>10}  "
                  f"{'N/A':>10}  "
                  f"{'N/A':>10}  "
                  f"✗ Failed")
    
    print("="*100)
    
    # Find best model
    successful_models = df[df['accuracy'].notna()]
    
    if len(successful_models) > 0:
        best_model = successful_models.loc[successful_models['accuracy'].idxmax()]
        print(f"\n🏆 BEST MODEL: Model {best_model['model_num']}")
        print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
        print(f"   Precision: {best_model['precision']:.4f}")
        print(f"   Recall: {best_model['recall']:.4f}")
        print(f"   F1-Score: {best_model['f1']:.4f}")
        print(f"   Results: {best_model['output_dir']}")
    
    # Save summary to CSV
    csv_path = output_dir / 'evaluation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary saved to: {csv_path}")
    
    # Save detailed text report
    report_lines = []
    report_lines.append("="*100)
    report_lines.append("MULTI-MODEL EVALUATION SUMMARY REPORT")
    report_lines.append("="*100)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Models Evaluated: {len(df)}")
    report_lines.append(f"Successful Evaluations: {len(successful_models)}")
    report_lines.append(f"Failed Evaluations: {len(df) - len(successful_models)}")
    report_lines.append("")
    report_lines.append("="*100)
    report_lines.append(f"{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    report_lines.append("="*100)
    
    for _, row in df.iterrows():
        if pd.notna(row.get('accuracy')):
            report_lines.append(
                f"{row['model_num']:<10} "
                f"{row['accuracy']*100:>10.2f}%  "
                f"{row['precision']:>10.4f}  "
                f"{row['recall']:>10.4f}  "
                f"{row['f1']:>10.4f}"
            )
        else:
            report_lines.append(
                f"{row['model_num']:<10} "
                f"{'FAILED':>11}  "
                f"{'N/A':>10}  "
                f"{'N/A':>10}  "
                f"{'N/A':>10}"
            )
    
    report_lines.append("="*100)
    
    if len(successful_models) > 0:
        report_lines.append("")
        report_lines.append("BEST MODEL")
        report_lines.append("-"*100)
        report_lines.append(f"Model Number: {best_model['model_num']}")
        report_lines.append(f"Model Path: {best_model['model_path']}")
        report_lines.append(f"Accuracy: {best_model['accuracy']*100:.2f}%")
        report_lines.append(f"Precision: {best_model['precision']:.4f}")
        report_lines.append(f"Recall: {best_model['recall']:.4f}")
        report_lines.append(f"F1-Score: {best_model['f1']:.4f}")
        report_lines.append(f"Output Directory: {best_model['output_dir']}")
        report_lines.append("")
        
        # Statistics
        report_lines.append("STATISTICS")
        report_lines.append("-"*100)
        report_lines.append(f"Mean Accuracy: {successful_models['accuracy'].mean()*100:.2f}%")
        report_lines.append(f"Std Accuracy: {successful_models['accuracy'].std()*100:.2f}%")
        report_lines.append(f"Mean Precision: {successful_models['precision'].mean():.4f}")
        report_lines.append(f"Mean Recall: {successful_models['recall'].mean():.4f}")
        report_lines.append(f"Mean F1-Score: {successful_models['f1'].mean():.4f}")
    
    report_lines.append("")
    report_lines.append("="*100)
    
    report_text = "\n".join(report_lines)
    
    # Save text report
    report_path = output_dir / 'evaluation_summary.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Detailed report saved to: {report_path}")
    
    # Generate comparison visualization
    if len(successful_models) > 0:
        plot_model_comparison(successful_models, output_dir)


def plot_model_comparison(df, output_dir):
    """Create comparison plots for all models"""
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'comparison_plots'
    viz_dir.mkdir(exist_ok=True)
    
    # Sort by model number
    df = df.sort_values('model_num')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = df['model_num'].values
    
    # Accuracy comparison
    axes[0, 0].bar(models, df['accuracy']*100, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0, 0].set_xlabel('Model Number', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold', fontsize=14)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=df['accuracy'].mean()*100, color='red', linestyle='--', 
                       label=f"Mean: {df['accuracy'].mean()*100:.2f}%")
    axes[0, 0].legend()
    
    # Precision comparison
    axes[0, 1].bar(models, df['precision'], color='lightcoral', edgecolor='darkred', alpha=0.7)
    axes[0, 1].set_xlabel('Model Number', fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_title('Precision Comparison', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axhline(y=df['precision'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['precision'].mean():.4f}")
    axes[0, 1].legend()
    
    # Recall comparison
    axes[1, 0].bar(models, df['recall'], color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    axes[1, 0].set_xlabel('Model Number', fontweight='bold')
    axes[1, 0].set_ylabel('Recall', fontweight='bold')
    axes[1, 0].set_title('Recall Comparison', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axhline(y=df['recall'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['recall'].mean():.4f}")
    axes[1, 0].legend()
    
    # F1-Score comparison
    axes[1, 1].bar(models, df['f1'], color='plum', edgecolor='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Model Number', fontweight='bold')
    axes[1, 1].set_ylabel('F1-Score', fontweight='bold')
    axes[1, 1].set_title('F1-Score Comparison', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axhline(y=df['f1'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['f1'].mean():.4f}")
    axes[1, 1].legend()
    
    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = viz_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    plt.close()
    
    # Create combined metrics plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['accuracy']*100, width, label='Accuracy (%)', color='skyblue')
    ax.bar(x - 0.5*width, df['precision']*100, width, label='Precision (%)', color='lightcoral')
    ax.bar(x + 0.5*width, df['recall']*100, width, label='Recall (%)', color='lightgreen')
    ax.bar(x + 1.5*width, df['f1']*100, width, label='F1-Score (%)', color='plum')
    
    ax.set_xlabel('Model Number', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('All Metrics Comparison', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    combined_plot_path = viz_dir / 'combined_metrics.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined metrics plot saved to: {combined_plot_path}")
    plt.close()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all ensemble models')
    parser.add_argument('--models_dir', type=str, 
                       default='video_classification_project/models_enhanced',
                       help='Directory containing model checkpoints')
    parser.add_argument('--h5_file', type=str, 
                       default='video_classification_project/features_enhanced/val_features_multiscale.h5', 
                       help='Path to H5 features file')
    parser.add_argument('--output_dir', type=str, 
                       default='video_classification_project/results_all_models',
                       help='Base output directory for all results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run evaluation for all models
    evaluate_all_models(
        models_dir=args.models_dir,
        h5_file=args.h5_file,
        base_output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()