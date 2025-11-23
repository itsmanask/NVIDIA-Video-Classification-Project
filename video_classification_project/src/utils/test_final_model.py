"""
FINAL MODEL TESTING SCRIPT
Test trained model on held-out test set with comprehensive evaluation

Features:
- Single model testing with TTA
- Ensemble model testing (individual + combined)
- Comprehensive metrics and visualizations
- Per-class performance analysis
- Confidence analysis
- Error analysis
- Separate results folders for each model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_recall_fscore_support,
    accuracy_score
)
import pandas as pd
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
        SuperEnhancedTemporalModel,
        EnhancedPreExtractedFeaturesDataset,
        EnhancedTemporalModelTrainer as CheckpointManager,
        collate_features
    )
    print("Successfully imported from model_train_new.py")
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error importing from model_train_new.py: {e}")
    print(f"Looking in: {data_dir}")
    sys.exit(1)


class SafePreExtractedFeaturesDataset(EnhancedPreExtractedFeaturesDataset):
    """Enhanced dataset with safe HDF5 file handling"""
    
    def __del__(self):
        """Safely close HDF5 file"""
        try:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                if hasattr(self.h5_file, 'id') and self.h5_file.id is not None:
                    self.h5_file.close()
        except Exception as e:
            # Silently ignore cleanup errors
            pass


class FinalModelTester:
    """Comprehensive testing on held-out test set"""
    
    def __init__(self, checkpoint_path, features_dir, output_dir, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print(f"FINAL MODEL TESTING ON TEST SET")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Features Dir: {features_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        self.model = None
        self.test_dataset = None
        self.category_mapping = None
        self.class_names = None
        self.num_classes = None
    
    def load_model_and_data(self, feature_suffix=''):
        """Load trained model and test dataset"""
        print(f"{'='*70}")
        print("LOADING MODEL AND TEST DATA")
        print(f"{'='*70}\n")
        
        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path.name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model configuration
        config = checkpoint.get('model_config', checkpoint.get('config', {}))
        
        feature_dim = config.get('feature_dim', 2048)
        hidden_dim = config.get('hidden_dim', 768)
        num_lstm_layers = config.get('num_lstm_layers', 4)
        num_attention_heads = config.get('num_attention_heads', 12)
        dropout = config.get('dropout', 0.4)
        bidirectional = config.get('bidirectional', True)
        
        print(f"\nModel Configuration:")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   LSTM layers: {num_lstm_layers}")
        print(f"   Attention heads: {num_attention_heads}")
        print(f"   Bidirectional: {bidirectional}")
        print(f"   Dropout: {dropout}")
        
        # Load test dataset
        test_feature_file = self.features_dir / f'test_features{feature_suffix}.h5'
        if not test_feature_file.exists():
            raise FileNotFoundError(f"Test features not found: {test_feature_file}")
        
        print(f"\nLoading test dataset: {test_feature_file.name}")
        self.test_dataset = SafePreExtractedFeaturesDataset(
            feature_file=test_feature_file,
            augment=False
        )
        
        self.category_mapping = self.test_dataset.category_mapping
        self.num_classes = len(self.category_mapping)
        self.class_names = [None] * self.num_classes
        
        for name, idx in self.category_mapping.items():
            self.class_names[idx] = name
        
        print(f"\nTest Dataset:")
        print(f"   Samples: {len(self.test_dataset):,}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Categories: {self.class_names}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        for class_name, class_id in sorted(self.category_mapping.items(), key=lambda x: x[1]):
            count = self.test_dataset.class_counts.get(class_id, 0)
            pct = 100 * count / len(self.test_dataset)
            print(f"   {class_name}: {count} ({pct:.1f}%)")
        
        # Initialize model
        print(f"\nInitializing model...")
        self.model = SuperEnhancedTemporalModel(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_lstm_layers=num_lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Display training info
        if 'epoch' in checkpoint:
            print(f"   Trained epochs: {checkpoint['epoch'] + 1}")
        if 'best_val_acc' in checkpoint:
            print(f"   Best val accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        print(f"\nModel and data loaded successfully!\n")
    
    def test_standard(self, batch_size=32):
        """Standard testing without augmentation"""
        print(f"{'='*70}")
        print("STANDARD TESTING")
        print(f"{'='*70}\n")
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_features,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        
        print(f"Running inference on {len(self.test_dataset)} samples...")
        with torch.no_grad():
            for features, labels, lengths in tqdm(test_loader, desc="Testing"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(features, lengths)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        print(f"\n{'='*70}")
        print("STANDARD TEST RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {f1.mean() * 100:.2f}%")
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision[i] * 100:.2f}%")
            print(f"      Recall:    {recall[i] * 100:.2f}%")
            print(f"      F1-Score:  {f1[i] * 100:.2f}%")
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def test_with_tta(self, batch_size=32):
        """Test with Test-Time Augmentation"""
        print(f"\n{'='*70}")
        print("TESTING WITH TTA (Test-Time Augmentation)")
        print(f"{'='*70}\n")
        
        tta_modes = [None, 'reverse', 'speed_up', 'speed_down']
        all_predictions = []
        
        for tta_mode in tta_modes:
            mode_name = tta_mode if tta_mode else 'original'
            print(f"\nTTA Mode: {mode_name}")
            
            # Create dataset with TTA
            tta_dataset = SafePreExtractedFeaturesDataset(
                feature_file=self.test_dataset.feature_file,
                augment=False,
                tta_mode=tta_mode
            )
            
            tta_loader = DataLoader(
                tta_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_features,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            mode_probs = []
            
            self.model.eval()
            with torch.no_grad():
                for features, labels, lengths in tqdm(tta_loader, desc=f"TTA {mode_name}"):
                    features = features.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    outputs = self.model(features, lengths)
                    probs = F.softmax(outputs, dim=1)
                    mode_probs.append(probs.cpu())
            
            mode_probs = torch.cat(mode_probs).numpy()
            all_predictions.append(mode_probs)
            
            # Clean up
            del tta_dataset
            del tta_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average predictions across all TTA modes
        avg_probs = np.mean(all_predictions, axis=0)
        tta_preds = np.argmax(avg_probs, axis=1)
        
        # Get ground truth labels
        labels = self.test_dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(labels, tta_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, tta_preds, average=None, zero_division=0
        )
        
        print(f"\n{'='*70}")
        print("TTA TEST RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {f1.mean() * 100:.2f}%")
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision[i] * 100:.2f}%")
            print(f"      Recall:    {recall[i] * 100:.2f}%")
            print(f"      F1-Score:  {f1[i] * 100:.2f}%")
        
        return {
            'predictions': tta_preds,
            'labels': labels,
            'probabilities': avg_probs,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def plot_confusion_matrix(self, results, title, filename):
        """Plot confusion matrix"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # Normalized version
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_title(f'{title} - Raw Counts', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        
        # Normalized
        sns.heatmap(
            cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1]
        )
        axes[1].set_title(f'{title} - Normalized', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
    
    def plot_metrics_comparison(self, standard_results, tta_results):
        """Compare standard vs TTA metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(self.num_classes)
        width = 0.35
        
        # Precision
        axes[0, 0].bar(x - width/2, standard_results['precision'], width, 
                      label='Standard', color='skyblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, tta_results['precision'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[0, 0].set_xlabel('Class', fontweight='bold')
        axes[0, 0].set_ylabel('Precision (%)', fontweight='bold')
        axes[0, 0].set_title('Precision Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[0, 1].bar(x - width/2, standard_results['recall'], width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, tta_results['recall'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('Class', fontweight='bold')
        axes[0, 1].set_ylabel('Recall (%)', fontweight='bold')
        axes[0, 1].set_title('Recall Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score
        axes[1, 0].bar(x - width/2, standard_results['f1'], width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[1, 0].bar(x + width/2, tta_results['f1'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[1, 0].set_xlabel('Class', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score (%)', fontweight='bold')
        axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Overall metrics
        metrics = ['Accuracy', 'Macro F1']
        standard_vals = [standard_results['accuracy'], standard_results['f1'].mean()]
        tta_vals = [tta_results['accuracy'], tta_results['f1'].mean()]
        
        x_pos = np.arange(len(metrics))
        axes[1, 1].bar(x_pos - width/2, standard_vals, width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, tta_vals, width,
                      label='TTA', color='coral', alpha=0.8)
        axes[1, 1].set_xlabel('Metric', fontweight='bold')
        axes[1, 1].set_ylabel('Score (%)', fontweight='bold')
        axes[1, 1].set_title('Overall Metrics', fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for ax in axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)
        
        plt.suptitle('Standard vs TTA Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: metrics_comparison.png")
        plt.close()
    
    def analyze_errors(self, results, title, filename):
        """Analyze prediction errors"""
        preds = results['predictions']
        labels = results['labels']
        probs = results['probabilities']
        
        errors = preds != labels
        error_indices = np.where(errors)[0]
        
        print(f"\n{'='*70}")
        print(f"ERROR ANALYSIS - {title}")
        print(f"{'='*70}")
        print(f"Total errors: {len(error_indices)} / {len(labels)} ({100*len(error_indices)/len(labels):.2f}%)")
        
        # Most common misclassifications
        print(f"\nMost Common Misclassifications:")
        misclass_pairs = {}
        for idx in error_indices:
            true_label = labels[idx]
            pred_label = preds[idx]
            pair = (self.class_names[true_label], self.class_names[pred_label])
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        for (true_class, pred_class), count in sorted_pairs[:10]:
            print(f"   {true_class} -> {pred_class}: {count} times")
        
        # Confidence analysis
        correct_confidence = probs[~errors].max(axis=1).mean()
        error_confidence = probs[errors].max(axis=1).mean()
        
        print(f"\nConfidence Analysis:")
        print(f"   Avg confidence (correct): {correct_confidence:.3f}")
        print(f"   Avg confidence (errors): {error_confidence:.3f}")
        print(f"   Confidence gap: {correct_confidence - error_confidence:.3f}")
        
        # Low confidence errors
        low_conf_threshold = 0.7
        low_conf_errors = error_indices[probs[errors].max(axis=1) < low_conf_threshold]
        print(f"   Low confidence errors (<{low_conf_threshold}): {len(low_conf_errors)} ({100*len(low_conf_errors)/len(error_indices):.1f}%)")
        
        # High confidence errors
        high_conf_threshold = 0.9
        high_conf_errors = error_indices[probs[errors].max(axis=1) >= high_conf_threshold]
        print(f"   High confidence errors (>={high_conf_threshold}): {len(high_conf_errors)} ({100*len(high_conf_errors)/len(error_indices):.1f}%)")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Error distribution by class
        error_by_class = [np.sum((labels[errors] == i)) for i in range(self.num_classes)]
        total_by_class = [np.sum(labels == i) for i in range(self.num_classes)]
        error_rate_by_class = [100 * e / (t + 1e-10) for e, t in zip(error_by_class, total_by_class)]
        
        axes[0].bar(self.class_names, error_rate_by_class, color='crimson', alpha=0.7)
        axes[0].set_xlabel('Class', fontweight='bold')
        axes[0].set_ylabel('Error Rate (%)', fontweight='bold')
        axes[0].set_title('Error Rate by Class', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, (bar, rate) in enumerate(zip(axes[0].containers[0], error_rate_by_class)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%\n({error_by_class[i]})',
                        ha='center', va='bottom', fontsize=9)
        
        # Confidence distribution
        axes[1].hist(probs[~errors].max(axis=1), bins=50, alpha=0.7, 
                    label='Correct', color='green', edgecolor='black')
        axes[1].hist(probs[errors].max(axis=1), bins=50, alpha=0.7,
                    label='Errors', color='red', edgecolor='black')
        axes[1].set_xlabel('Confidence', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].set_title('Confidence Distribution', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
    
    def generate_report(self, standard_results, tta_results):
        """Generate comprehensive test report"""
        report = []
        report.append("="*80)
        report.append("FINAL MODEL TEST REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Checkpoint: {self.checkpoint_path}")
        report.append(f"Test samples: {len(self.test_dataset):,}")
        report.append("")
        
        # Standard results
        report.append("STANDARD TESTING")
        report.append("-"*80)
        report.append(f"Accuracy: {standard_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {standard_results['f1'].mean():.2f}%")
        report.append("")
        report.append("Per-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            report.append(f"   {class_name}:")
            report.append(f"      Precision: {standard_results['precision'][i]:.2f}%")
            report.append(f"      Recall:    {standard_results['recall'][i]:.2f}%")
            report.append(f"      F1-Score:  {standard_results['f1'][i]:.2f}%")
        report.append("")
        
        # TTA results
        report.append("TTA TESTING")
        report.append("-"*80)
        report.append(f"Accuracy: {tta_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {tta_results['f1'].mean():.2f}%")
        report.append("")
        report.append("Per-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            report.append(f"   {class_name}:")
            report.append(f"      Precision: {tta_results['precision'][i]:.2f}%")
            report.append(f"      Recall:    {tta_results['recall'][i]:.2f}%")
            report.append(f"      F1-Score:  {tta_results['f1'][i]:.2f}%")
        report.append("")
        
        # Improvement
        report.append("TTA IMPROVEMENT")
        report.append("-"*80)
        acc_improvement = tta_results['accuracy'] - standard_results['accuracy']
        f1_improvement = tta_results['f1'].mean() - standard_results['f1'].mean()
        report.append(f"Accuracy improvement: +{acc_improvement:.2f}%")
        report.append(f"Macro F1 improvement: +{f1_improvement:.2f}%")
        report.append("")
        
        # Best/Worst classes
        report.append("CLASS ANALYSIS (TTA)")
        report.append("-"*80)
        best_idx = np.argmax(tta_results['f1'])
        worst_idx = np.argmin(tta_results['f1'])
        report.append(f"Best performing class:  {self.class_names[best_idx]} (F1: {tta_results['f1'][best_idx]:.2f}%)")
        report.append(f"Worst performing class: {self.class_names[worst_idx]} (F1: {tta_results['f1'][worst_idx]:.2f}%)")
        report.append("")
        
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'test_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n{report_text}")
        print(f"\nReport saved to: {report_path}")
        
        # Save JSON results
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(self.checkpoint_path),
            'test_samples': len(self.test_dataset),
            'standard': {
                'accuracy': float(standard_results['accuracy']),
                'macro_f1': float(standard_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(standard_results['precision'][i]),
                        'recall': float(standard_results['recall'][i]),
                        'f1': float(standard_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            },
            'tta': {
                'accuracy': float(tta_results['accuracy']),
                'macro_f1': float(tta_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(tta_results['precision'][i]),
                        'recall': float(tta_results['recall'][i]),
                        'f1': float(tta_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            },
            'improvement': {
                'accuracy': float(acc_improvement),
                'macro_f1': float(f1_improvement)
            }
        }
        
        json_path = self.output_dir / 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
    
    def run_complete_test(self, batch_size=32, feature_suffix=''):
        """Run complete testing pipeline"""
        print(f"\n{'='*80}")
        print("STARTING COMPLETE TEST EVALUATION")
        print(f"{'='*80}\n")
        
        # Load model and data
        self.load_model_and_data(feature_suffix=feature_suffix)
        
        # Standard testing
        print(f"\n{'='*80}")
        print("PHASE 1: STANDARD TESTING")
        print(f"{'='*80}")
        standard_results = self.test_standard(batch_size=batch_size)
        
        # TTA testing
        print(f"\n{'='*80}")
        print("PHASE 2: TTA TESTING")
        print(f"{'='*80}")
        tta_results = self.test_with_tta(batch_size=batch_size)
        
        # Visualizations
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        self.plot_confusion_matrix(
            standard_results, 
            "Standard Testing","confusion_matrix_standard.png"
        )
        
        self.plot_confusion_matrix(
            tta_results,
            "TTA Testing",
            "confusion_matrix_tta.png"
        )
        
        self.plot_metrics_comparison(standard_results, tta_results)
        
        self.analyze_errors(standard_results, "Standard", "error_analysis_standard.png")
        self.analyze_errors(tta_results, "TTA", "error_analysis_tta.png")
        
        # Generate report
        print(f"\n{'='*80}")
        print("GENERATING REPORT")
        print(f"{'='*80}")
        self.generate_report(standard_results, tta_results)
        
        print(f"\n{'='*80}")
        print("TESTING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal Results:")
        print(f"   Standard Accuracy: {standard_results['accuracy']:.2f}%")
        print(f"   TTA Accuracy:      {tta_results['accuracy']:.2f}%")
        print(f"   Improvement:       +{tta_results['accuracy'] - standard_results['accuracy']:.2f}%")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"   - test_report.txt")
        print(f"   - test_results.json")
        print(f"   - confusion_matrix_standard.png")
        print(f"   - confusion_matrix_tta.png")
        print(f"   - metrics_comparison.png")
        print(f"   - error_analysis_standard.png")
        print(f"   - error_analysis_tta.png")
        print()
        
        return standard_results, tta_results


class EnsembleTester:
    """Test ensemble of models with individual model testing"""
    
    def __init__(self, checkpoint_paths, features_dir, output_dir, device='cuda'):
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE MODEL TESTING ({len(self.checkpoint_paths)} models)")
        print(f"{'='*80}")
        for i, path in enumerate(self.checkpoint_paths):
            print(f"   Model {i}: {path.name}")
        print(f"Features Dir: {features_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        self.models = []
        self.test_dataset = None
        self.category_mapping = None
        self.class_names = None
        self.num_classes = None
        self.individual_results = []
    
    def load_models_and_data(self, feature_suffix=''):
        """Load all ensemble models and test data"""
        print(f"{'='*70}")
        print("LOADING ENSEMBLE MODELS")
        print(f"{'='*70}\n")
        
        # Load test dataset (same for all models)
        test_feature_file = self.features_dir / f'test_features{feature_suffix}.h5'
        if not test_feature_file.exists():
            raise FileNotFoundError(f"Test features not found: {test_feature_file}")
        
        print(f"Loading test dataset: {test_feature_file.name}")
        self.test_dataset = SafePreExtractedFeaturesDataset(
            feature_file=test_feature_file,
            augment=False
        )
        
        self.category_mapping = self.test_dataset.category_mapping
        self.num_classes = len(self.category_mapping)
        self.class_names = [None] * self.num_classes
        
        for name, idx in self.category_mapping.items():
            self.class_names[idx] = name
        
        print(f"Test samples: {len(self.test_dataset):,}")
        print(f"Classes: {self.class_names}\n")
        
        # Load all models
        for i, checkpoint_path in enumerate(self.checkpoint_paths):
            print(f"Loading model {i}/{len(self.checkpoint_paths)-1}: {checkpoint_path.name}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('model_config', checkpoint.get('config', {}))
            
            model = SuperEnhancedTemporalModel(
                feature_dim=config.get('feature_dim', 2048),
                hidden_dim=config.get('hidden_dim', 768),
                num_classes=self.num_classes,
                num_lstm_layers=config.get('num_lstm_layers', 4),
                num_attention_heads=config.get('num_attention_heads', 12),
                dropout=config.get('dropout', 0.4),
                bidirectional=config.get('bidirectional', True)
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            
            if 'best_val_acc' in checkpoint:
                print(f"   Val accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        print(f"\nAll {len(self.models)} models loaded!\n")
    
    def test_individual_model(self, model_idx, batch_size=32, use_tta=False):
        """Test individual model from ensemble"""
        model = self.models[model_idx]
        checkpoint_name = self.checkpoint_paths[model_idx].stem
        
        print(f"\n{'='*70}")
        print(f"TESTING MODEL {model_idx}: {checkpoint_name}")
        print(f"Mode: {'TTA' if use_tta else 'STANDARD'}")
        print(f"{'='*70}\n")
        
        if use_tta:
            tta_modes = [None, 'reverse', 'speed_up', 'speed_down']
        else:
            tta_modes = [None]
        
        model_predictions = []
        
        for tta_mode in tta_modes:
            if use_tta:
                mode_name = tta_mode if tta_mode else 'original'
                print(f"   TTA mode: {mode_name}")
            
            # Create dataset with TTA
            tta_dataset = SafePreExtractedFeaturesDataset(
                feature_file=self.test_dataset.feature_file,
                augment=False,
                tta_mode=tta_mode
            )
            
            tta_loader = DataLoader(
                tta_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_features,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            mode_probs = []
            
            model.eval()
            with torch.no_grad():
                for features, labels, lengths in tqdm(tta_loader, desc=f"Model {model_idx}"):
                    features = features.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    outputs = model(features, lengths)
                    probs = F.softmax(outputs, dim=1)
                    mode_probs.append(probs.cpu())
            
            mode_probs = torch.cat(mode_probs).numpy()
            model_predictions.append(mode_probs)
            
            # Clean up
            del tta_dataset
            del tta_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average predictions
        avg_probs = np.mean(model_predictions, axis=0)
        preds = np.argmax(avg_probs, axis=1)
        labels = self.test_dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        print(f"\nModel {model_idx} Results:")
        print(f"   Accuracy: {accuracy * 100:.2f}%")
        print(f"   Macro F1: {f1.mean() * 100:.2f}%")
        
        return {
            'model_idx': model_idx,
            'checkpoint_name': checkpoint_name,
            'predictions': preds,
            'labels': labels,
            'probabilities': avg_probs,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def test_all_individual_models(self, batch_size=32, feature_suffix=''):
        """Test each ensemble model individually"""
        print(f"\n{'='*80}")
        print("TESTING INDIVIDUAL MODELS")
        print(f"{'='*80}\n")
        
        self.individual_results = []
        
        for model_idx in range(len(self.models)):
            # Create separate output directory for this model
            model_output_dir = self.output_dir / f'model_{model_idx}'
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test without TTA
            print(f"\n{'='*70}")
            print(f"MODEL {model_idx} - STANDARD TESTING")
            print(f"{'='*70}")
            standard_results = self.test_individual_model(model_idx, batch_size, use_tta=False)
            
            # Test with TTA
            print(f"\n{'='*70}")
            print(f"MODEL {model_idx} - TTA TESTING")
            print(f"{'='*70}")
            tta_results = self.test_individual_model(model_idx, batch_size, use_tta=True)
            
            # Save results for this model
            self._save_individual_model_results(
                model_idx, standard_results, tta_results, model_output_dir
            )
            
            self.individual_results.append({
                'standard': standard_results,
                'tta': tta_results
            })
            
            print(f"\n{'='*70}")
            print(f"MODEL {model_idx} COMPLETE")
            print(f"   Standard: {standard_results['accuracy']:.2f}%")
            print(f"   TTA:      {tta_results['accuracy']:.2f}%")
            print(f"   Results saved to: {model_output_dir}")
            print(f"{'='*70}\n")
    
    def _save_individual_model_results(self, model_idx, standard_results, tta_results, output_dir):
        """Save individual model results"""
        # Confusion matrices
        self._plot_confusion_matrix(
            standard_results, 
            f"Model {model_idx} - Standard",
            output_dir / "confusion_matrix_standard.png"
        )
        
        self._plot_confusion_matrix(
            tta_results,
            f"Model {model_idx} - TTA",
            output_dir / "confusion_matrix_tta.png"
        )
        
        # Metrics comparison
        self._plot_individual_metrics(standard_results, tta_results, model_idx, output_dir)
        
        # Text report
        self._generate_individual_report(model_idx, standard_results, tta_results, output_dir)
        
        # JSON results
        json_results = {
            'model_idx': model_idx,
            'checkpoint': standard_results['checkpoint_name'],
            'timestamp': datetime.now().isoformat(),
            'standard': {
                'accuracy': float(standard_results['accuracy']),
                'macro_f1': float(standard_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(standard_results['precision'][i]),
                        'recall': float(standard_results['recall'][i]),
                        'f1': float(standard_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            },
            'tta': {
                'accuracy': float(tta_results['accuracy']),
                'macro_f1': float(tta_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(tta_results['precision'][i]),
                        'recall': float(tta_results['recall'][i]),
                        'f1': float(tta_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            }
        }
        
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _plot_individual_metrics(self, standard_results, tta_results, model_idx, output_dir):
        """Plot metrics for individual model"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(self.num_classes)
        width = 0.35
        
        # Precision
        axes[0, 0].bar(x - width/2, standard_results['precision'], width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, tta_results['precision'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[0, 0].set_xlabel('Class', fontweight='bold')
        axes[0, 0].set_ylabel('Precision (%)', fontweight='bold')
        axes[0, 0].set_title(f'Model {model_idx} - Precision', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[0, 1].bar(x - width/2, standard_results['recall'], width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, tta_results['recall'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('Class', fontweight='bold')
        axes[0, 1].set_ylabel('Recall (%)', fontweight='bold')
        axes[0, 1].set_title(f'Model {model_idx} - Recall', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score
        axes[1, 0].bar(x - width/2, standard_results['f1'], width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[1, 0].bar(x + width/2, tta_results['f1'], width,
                      label='TTA', color='coral', alpha=0.8)
        axes[1, 0].set_xlabel('Class', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score (%)', fontweight='bold')
        axes[1, 0].set_title(f'Model {model_idx} - F1-Score', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Overall
        metrics = ['Accuracy', 'Macro F1']
        standard_vals = [standard_results['accuracy'], standard_results['f1'].mean()]
        tta_vals = [tta_results['accuracy'], tta_results['f1'].mean()]
        
        x_pos = np.arange(len(metrics))
        axes[1, 1].bar(x_pos - width/2, standard_vals, width,
                      label='Standard', color='skyblue', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, tta_vals, width,
                      label='TTA', color='coral', alpha=0.8)
        axes[1, 1].set_xlabel('Metric', fontweight='bold')
        axes[1, 1].set_ylabel('Score (%)', fontweight='bold')
        axes[1, 1].set_title(f'Model {model_idx} - Overall', fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for ax in axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3)
        
        plt.suptitle(f'Model {model_idx} Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_individual_report(self, model_idx, standard_results, tta_results, output_dir):
        """Generate report for individual model"""
        report = []
        report.append("="*80)
        report.append(f"MODEL {model_idx} TEST REPORT")
        report.append("="*80)
        report.append(f"Checkpoint: {standard_results['checkpoint_name']}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("STANDARD TESTING")
        report.append("-"*80)
        report.append(f"Accuracy: {standard_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {standard_results['f1'].mean():.2f}%")
        report.append("")
        
        report.append("TTA TESTING")
        report.append("-"*80)
        report.append(f"Accuracy: {tta_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {tta_results['f1'].mean():.2f}%")
        report.append("")
        
        report.append("IMPROVEMENT")
        report.append("-"*80)
        acc_imp = tta_results['accuracy'] - standard_results['accuracy']
        report.append(f"Accuracy: +{acc_imp:.2f}%")
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        report_path = output_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
    
    def test_ensemble(self, batch_size=32, use_tta=False):
        """Test ensemble of models"""
        print(f"{'='*70}")
        print(f"TESTING ENSEMBLE ({'WITH TTA' if use_tta else 'STANDARD'})")
        print(f"{'='*70}\n")
        
        if use_tta:
            tta_modes = [None, 'reverse', 'speed_up', 'speed_down']
        else:
            tta_modes = [None]
        
        # Collect predictions from all models
        all_model_predictions = []
        
        for model_idx, model in enumerate(self.models):
            print(f"\nModel {model_idx}/{len(self.models)-1}")
            
            model_tta_predictions = []
            
            for tta_mode in tta_modes:
                if use_tta:
                    mode_name = tta_mode if tta_mode else 'original'
                    print(f"   TTA mode: {mode_name}")
                
                tta_dataset = SafePreExtractedFeaturesDataset(
                    feature_file=self.test_dataset.feature_file,
                    augment=False,
                    tta_mode=tta_mode
                )
                
                tta_loader = DataLoader(
                    tta_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_features,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                mode_probs = []
                
                model.eval()
                with torch.no_grad():
                    for features, labels, lengths in tqdm(tta_loader, desc=f"Model {model_idx}"):
                        features = features.to(self.device)
                        lengths = lengths.to(self.device)
                        
                        outputs = model(features, lengths)
                        probs = F.softmax(outputs, dim=1)
                        mode_probs.append(probs.cpu())
                
                mode_probs = torch.cat(mode_probs).numpy()
                model_tta_predictions.append(mode_probs)
                
                # Clean up
                del tta_dataset
                del tta_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average TTA modes for this model
            model_avg_probs = np.mean(model_tta_predictions, axis=0)
            all_model_predictions.append(model_avg_probs)
        
        # Average across all models
        ensemble_probs = np.mean(all_model_predictions, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        labels = self.test_dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(labels, ensemble_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, ensemble_preds, average=None, zero_division=0
        )
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE RESULTS ({'WITH TTA' if use_tta else 'STANDARD'})")
        print(f"{'='*70}")
        print(f"Models: {len(self.models)}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {f1.mean() * 100:.2f}%")
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision[i] * 100:.2f}%")
            print(f"      Recall:    {recall[i] * 100:.2f}%")
            print(f"      F1-Score:  {f1[i] * 100:.2f}%")
        
        return {
            'predictions': ensemble_preds,
            'labels': labels,
            'probabilities': ensemble_probs,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def run_complete_ensemble_test(self, batch_size=32, feature_suffix=''):
        """Run complete ensemble testing"""
        print(f"\n{'='*80}")
        print("STARTING COMPLETE ENSEMBLE TESTING")
        print(f"{'='*80}\n")
        
        # Load models
        self.load_models_and_data(feature_suffix=feature_suffix)
        
        # Test individual models first
        print(f"\n{'='*80}")
        print("PHASE 1: TESTING INDIVIDUAL MODELS")
        print(f"{'='*80}")
        self.test_all_individual_models(batch_size=batch_size, feature_suffix=feature_suffix)
        
        # Create ensemble results directory
        ensemble_dir = self.output_dir / 'ensemble_combined'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Test ensemble without TTA
        print(f"\n{'='*80}")
        print("PHASE 2: ENSEMBLE (STANDARD)")
        print(f"{'='*80}")
        standard_results = self.test_ensemble(batch_size=batch_size, use_tta=False)
        
        # Test ensemble with TTA
        print(f"\n{'='*80}")
        print("PHASE 3: ENSEMBLE + TTA")
        print(f"{'='*80}")
        tta_results = self.test_ensemble(batch_size=batch_size, use_tta=True)
        
        # Visualizations for ensemble
        print(f"\n{'='*80}")
        print("GENERATING ENSEMBLE VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        self._plot_confusion_matrix(standard_results, "Ensemble (Standard)", 
                                    ensemble_dir / "confusion_matrix_standard.png")
        self._plot_confusion_matrix(tta_results, "Ensemble + TTA",
                                    ensemble_dir / "confusion_matrix_tta.png")
        
        # Generate ensemble report
        self._generate_ensemble_report(standard_results, tta_results, ensemble_dir)
        
        # Generate summary across all models
        self._generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ENSEMBLE TESTING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal Results:")
        print(f"   Ensemble (Standard): {standard_results['accuracy']:.2f}%")
        print(f"   Ensemble + TTA:      {tta_results['accuracy']:.2f}%")
        print(f"   Improvement:         +{tta_results['accuracy'] - standard_results['accuracy']:.2f}%")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"   - Individual models: model_0/, model_1/, ...")
        print(f"   - Ensemble results: ensemble_combined/")
        print(f"   - Summary: summary_report.txt")
        print()
        
        return standard_results, tta_results
    
    def _plot_confusion_matrix(self, results, title, filepath):
        """Plot confusion matrix"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title(f'{title} - Raw Counts', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1])
        axes[1].set_title(f'{title} - Normalized', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filepath.name}")
        plt.close()
    
    def _generate_ensemble_report(self, standard_results, tta_results, output_dir):
        """Generate ensemble test report"""
        report = []
        report.append("="*80)
        report.append("ENSEMBLE TEST REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of models: {len(self.models)}")
        report.append(f"Test samples: {len(self.test_dataset):,}")
        report.append("")
        
        report.append("ENSEMBLE (STANDARD)")
        report.append("-"*80)
        report.append(f"Accuracy: {standard_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {standard_results['f1'].mean():.2f}%")
        report.append("")
        
        report.append("ENSEMBLE + TTA")
        report.append("-"*80)
        report.append(f"Accuracy: {tta_results['accuracy']:.2f}%")
        report.append(f"Macro F1: {tta_results['f1'].mean():.2f}%")
        report.append("")
        
        report.append("IMPROVEMENT")
        report.append("-"*80)
        acc_improvement = tta_results['accuracy'] - standard_results['accuracy']
        report.append(f"TTA improvement: +{acc_improvement:.2f}%")
        report.append("")
        
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = output_dir / 'ensemble_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nEnsemble report saved to: {report_path}")
        
        # Save JSON
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.models),
            'test_samples': len(self.test_dataset),
            'standard': {
                'accuracy': float(standard_results['accuracy']),
                'macro_f1': float(standard_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(standard_results['precision'][i]),
                        'recall': float(standard_results['recall'][i]),
                        'f1': float(standard_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            },
            'tta': {
                'accuracy': float(tta_results['accuracy']),
                'macro_f1': float(tta_results['f1'].mean()),
                'per_class': {
                    self.class_names[i]: {
                        'precision': float(tta_results['precision'][i]),
                        'recall': float(tta_results['recall'][i]),
                        'f1': float(tta_results['f1'][i])
                    }
                    for i in range(self.num_classes)
                }
            },
            'improvement': float(acc_improvement)
        }
        
        json_path = output_dir / 'ensemble_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
    
    def _generate_summary_report(self):
        """Generate summary report comparing all models"""
        report = []
        report.append("="*80)
        report.append("COMPLETE TESTING SUMMARY")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total models tested: {len(self.models)}")
        report.append(f"Test samples: {len(self.test_dataset):,}")
        report.append("")
        
        report.append("INDIVIDUAL MODEL RESULTS")
        report.append("="*80)
        report.append("")
        
        # Standard results for each model
        report.append("STANDARD TESTING:")
        report.append("-"*80)
        for i, result in enumerate(self.individual_results):
            std = result['standard']
            report.append(f"Model {i}: {std['accuracy']:.2f}% (F1: {std['f1'].mean():.2f}%)")
        report.append("")
        
        # TTA results for each model
        report.append("TTA TESTING:")
        report.append("-"*80)
        for i, result in enumerate(self.individual_results):
            tta = result['tta']
            report.append(f"Model {i}: {tta['accuracy']:.2f}% (F1: {tta['f1'].mean():.2f}%)")
        report.append("")
        
        # Best performing model
        best_model_idx = np.argmax([r['tta']['accuracy'] for r in self.individual_results])
        best_acc = self.individual_results[best_model_idx]['tta']['accuracy']
        report.append(f"Best Individual Model: Model {best_model_idx} ({best_acc:.2f}%)")
        report.append("")
        
        report.append("="*80)
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save summary
        summary_path = self.output_dir / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n{report_text}")
        print(f"\nSummary report saved to: {summary_path}")
        
        # Save summary JSON
        summary_json = {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.models),
            'test_samples': len(self.test_dataset),
            'individual_models': [
                {
                    'model_idx': i,
                    'checkpoint': result['standard']['checkpoint_name'],
                    'standard_accuracy': float(result['standard']['accuracy']),
                    'tta_accuracy': float(result['tta']['accuracy']),
                    'standard_f1': float(result['standard']['f1'].mean()),
                    'tta_f1': float(result['tta']['f1'].mean())
                }
                for i, result in enumerate(self.individual_results)
            ],
            'best_model': {
                'model_idx': int(best_model_idx),
                'accuracy': float(best_acc)
            }
        }
        
        json_path = self.output_dir / 'summary_results.json'
        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)


def main():
    """Main execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (or first checkpoint for ensemble)')
    parser.add_argument('--features_dir', type=str,
                       default='video_classification_project/features_enhanced',
                       help='Directory containing test features')
    parser.add_argument('--output_dir', type=str,
                       default='video_classification_project/test_results',
                       help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--feature_suffix', type=str, default='',
                       help='Feature file suffix (e.g., "_multiscale")')
    parser.add_argument('--ensemble', nargs='+', type=str, default=None,
                       help='List of checkpoint paths for ensemble testing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if ensemble mode
    if args.ensemble:
        print(f"\nENSEMBLE MODE: Testing {len(args.ensemble)} models")
        
        tester = EnsembleTester(
            checkpoint_paths=args.ensemble,
            features_dir=args.features_dir,
            output_dir=args.output_dir,
            device=args.device
        )
        
        standard_results, tta_results = tester.run_complete_ensemble_test(
            batch_size=args.batch_size,
            feature_suffix=args.feature_suffix
        )
        
    else:
        print(f"\nSINGLE MODEL MODE")
        
        tester = FinalModelTester(
            checkpoint_path=args.checkpoint,
            features_dir=args.features_dir,
            output_dir=args.output_dir,
            device=args.device
        )
        
        standard_results, tta_results = tester.run_complete_test(
            batch_size=args.batch_size,
            feature_suffix=args.feature_suffix
        )
    
    print(f"\n{'='*80}")
    print("ALL TESTING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Auto mode if no arguments
    if len(sys.argv) == 1:
        print("="*80)
        print("FINAL MODEL TESTING - AUTO MODE")
        print("="*80)
        print("\nSearching for checkpoints and test data...\n")
        
        # Search for checkpoints
        possible_dirs = [
            Path('video_classification_project/models_enhanced'),
            Path('models_enhanced'),
            Path('video_classification_project/models'),
            Path('models')
        ]
        
        checkpoint = None
        ensemble_checkpoints = []
        
        for model_dir in possible_dirs:
            if model_dir.exists():
                # Look for ensemble models
                ensemble_models = sorted(model_dir.glob('best_ensemble_model_*.pt'))
                if ensemble_models:
                    ensemble_checkpoints = [str(p) for p in ensemble_models]
                    print(f"Found {len(ensemble_checkpoints)} ensemble models in {model_dir}")
                    for i, ckpt in enumerate(ensemble_checkpoints):
                        print(f"   Model {i}: {Path(ckpt).name}")
                    break
                
                # Look for best single model
                best_model = model_dir / 'best_single_model.pt'
                if best_model.exists():
                    checkpoint = best_model
                    print(f"Found single model checkpoint: {checkpoint}")
                    break
        
        if not checkpoint and not ensemble_checkpoints:
            print("No checkpoint found!")
            print("\nPlease run with explicit checkpoint path:")
            print("   python test_final_model.py --checkpoint path/to/checkpoint.pt")
            print("\nOr for ensemble:")
            print("   python test_final_model.py --ensemble model1.pt model2.pt model3.pt")
            sys.exit(1)
        
        # Search for features
        features_dirs = [
            Path('video_classification_project/features_enhanced'),
            Path('features_enhanced'),
            Path('video_classification_project/features'),
            Path('features')
        ]
        
        features_dir = None
        feature_suffix = ''
        for fdir in features_dirs:
            if (fdir / 'test_features.h5').exists():
                features_dir = fdir
                print(f"Found test features: {fdir / 'test_features.h5'}")
                break
            elif (fdir / 'test_features_multiscale.h5').exists():
                features_dir = fdir
                feature_suffix = '_multiscale'
                print(f"Found test features: {fdir / 'test_features_multiscale.h5'}")
                break
        
        if not features_dir:
            print("No test features found!")
            print("\nPlease run Stage 1 (feature extraction) first")
            sys.exit(1)
        
        # Determine output directory
        output_dir = Path('video_classification_project/test_results')
        
        # Run testing
        print(f"\n{'='*80}")
        print("STARTING AUTO TEST")
        print(f"{'='*80}\n")
        
        try:
            if ensemble_checkpoints:
                print(f"TESTING ENSEMBLE OF {len(ensemble_checkpoints)} MODELS\n")
                
                tester = EnsembleTester(
                    checkpoint_paths=ensemble_checkpoints,
                    features_dir=features_dir,
                    output_dir=output_dir,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                standard_results, tta_results = tester.run_complete_ensemble_test(
                    batch_size=32,
                    feature_suffix=feature_suffix
                )
            else:
                print(f"TESTING SINGLE MODEL\n")
                
                tester = FinalModelTester(
                    checkpoint_path=checkpoint,
                    features_dir=features_dir,
                    output_dir=output_dir,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                standard_results, tta_results = tester.run_complete_test(
                    batch_size=32,
                    feature_suffix=feature_suffix
                )
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        main()