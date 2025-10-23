"""
FINAL MODEL TESTING SCRIPT
Test trained model on held-out test set with comprehensive evaluation

Features:
- Single model testing with TTA
- Ensemble model testing
- Comprehensive metrics and visualizations
- Per-class performance analysis
- Confidence analysis
- Error analysis
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
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
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
        self.test_dataset = EnhancedPreExtractedFeaturesDataset(
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
        
        print(f"\n✅ Model and data loaded successfully!\n")
    
    def test_standard(self, batch_size=32):
        """Standard testing without augmentation"""
        print(f"{'='*70}")
        print("STANDARD TESTING")
        print(f"{'='*70}\n")
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_features,
            pin_memory=True
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
            tta_dataset = EnhancedPreExtractedFeaturesDataset(
                feature_file=self.test_dataset.feature_file,
                augment=False,
                tta_mode=tta_mode
            )
            
            tta_loader = DataLoader(
                tta_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_features,
                pin_memory=True
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
            print(f"   {true_class} → {pred_class}: {count} times")
        
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
        
        # High confidence errors (concerning!)
        high_conf_threshold = 0.9
        high_conf_errors = error_indices[probs[errors].max(axis=1) >= high_conf_threshold]
        print(f"   High confidence errors (≥{high_conf_threshold}): {len(high_conf_errors)} ({100*len(high_conf_errors)/len(error_indices):.1f}%)")
        
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
            "Standard Testing", 
            "confusion_matrix_standard.png"
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
        print("✅ TESTING COMPLETE!")
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
    """Test ensemble of models"""
    
    def __init__(self, checkpoint_paths, features_dir, output_dir, device='cuda'):
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE MODEL TESTING ({len(checkpoint_paths)} models)")
        print(f"{'='*80}")
        for i, path in enumerate(checkpoint_paths):
            print(f"   Model {i+1}: {path.name}")
        print(f"Features Dir: {features_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        self.models = []
        self.test_dataset = None
        self.category_mapping = None
        self.class_names = None
        self.num_classes = None
    
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
        self.test_dataset = EnhancedPreExtractedFeaturesDataset(
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
            print(f"Loading model {i+1}/{len(self.checkpoint_paths)}: {checkpoint_path.name}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
        
        print(f"\n✅ All {len(self.models)} models loaded!\n")
    
    def test_ensemble(self, batch_size=32, use_tta=False):
        """Test ensemble of models"""
        print(f"{'='*70}")
        print(f"TESTING ENSEMBLE ({'WITH TTA' if use_tta else 'STANDARD'})")
        print(f"{'='*70}\n")
        
        if use_tta:
            tta_modes = [None, 'reverse', 'speed_up', 'speed_down']
        else:
            tta_modes = [None]
        
        # Collect predictions from all models and TTA modes
        all_model_predictions = []
        
        for model_idx, model in enumerate(self.models):
            print(f"\nModel {model_idx + 1}/{len(self.models)}")
            
            model_tta_predictions = []
            
            for tta_mode in tta_modes:
                if use_tta:
                    mode_name = tta_mode if tta_mode else 'original'
                    print(f"   TTA mode: {mode_name}")
                
                # Create dataset with TTA
                tta_dataset = EnhancedPreExtractedFeaturesDataset(
                    feature_file=self.test_dataset.feature_file,
                    augment=False,
                    tta_mode=tta_mode
                )
                
                tta_loader = DataLoader(
                    tta_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    collate_fn=collate_features,
                    pin_memory=True
                )
                
                mode_probs = []
                
                model.eval()
                with torch.no_grad():
                    for features, labels, lengths in tqdm(tta_loader, desc=f"Model {model_idx+1}"):
                        features = features.to(self.device)
                        lengths = lengths.to(self.device)
                        
                        outputs = model(features, lengths)
                        probs = F.softmax(outputs, dim=1)
                        mode_probs.append(probs.cpu())
                
                mode_probs = torch.cat(mode_probs).numpy()
                model_tta_predictions.append(mode_probs)
            
            # Average TTA modes for this model
            model_avg_probs = np.mean(model_tta_predictions, axis=0)
            all_model_predictions.append(model_avg_probs)
        
        # Average across all models
        ensemble_probs = np.mean(all_model_predictions, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Get labels
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
        print("STARTING ENSEMBLE TESTING")
        print(f"{'='*80}\n")
        
        # Load models
        self.load_models_and_data(feature_suffix=feature_suffix)
        
        # Test without TTA
        print(f"\n{'='*80}")
        print("PHASE 1: ENSEMBLE (STANDARD)")
        print(f"{'='*80}")
        standard_results = self.test_ensemble(batch_size=batch_size, use_tta=False)
        
        # Test with TTA
        print(f"\n{'='*80}")
        print("PHASE 2: ENSEMBLE + TTA")
        print(f"{'='*80}")
        tta_results = self.test_ensemble(batch_size=batch_size, use_tta=True)
        
        # Visualizations
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        # Confusion matrices
        self._plot_confusion_matrix(standard_results, "Ensemble (Standard)", 
                                    "ensemble_confusion_matrix_standard.png")
        self._plot_confusion_matrix(tta_results, "Ensemble + TTA",
                                    "ensemble_confusion_matrix_tta.png")
        
        # Generate report
        self._generate_ensemble_report(standard_results, tta_results)
        
        print(f"\n{'='*80}")
        print("✅ ENSEMBLE TESTING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal Results:")
        print(f"   Ensemble (Standard): {standard_results['accuracy']:.2f}%")
        print(f"   Ensemble + TTA:      {tta_results['accuracy']:.2f}%")
        print(f"   Improvement:         +{tta_results['accuracy'] - standard_results['accuracy']:.2f}%")
        print(f"\nAll results saved to: {self.output_dir}")
        print()
        
        return standard_results, tta_results
    
    def _plot_confusion_matrix(self, results, title, filename):
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
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()
    
    def _generate_ensemble_report(self, standard_results, tta_results):
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
        report_path = self.output_dir / 'ensemble_test_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n{report_text}")
        print(f"Report saved to: {report_path}")
        
        # Save JSON
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.models),
            'test_samples': len(self.test_dataset),
            'standard': {
                'accuracy': float(standard_results['accuracy']),
                'macro_f1': float(standard_results['f1'].mean())
            },
            'tta': {
                'accuracy': float(tta_results['accuracy']),
                'macro_f1': float(tta_results['f1'].mean())
            },
            'improvement': float(acc_improvement)
        }
        
        json_path = self.output_dir / 'ensemble_test_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

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
        print(f"\n🔹 ENSEMBLE MODE: Testing {len(args.ensemble)} models")
        
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
        print(f"\n🔹 SINGLE MODEL MODE")
        
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
    print("🎉 ALL TESTING COMPLETE!")
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
        for model_dir in possible_dirs:
            if model_dir.exists():
                # Look for best model
                best_model = model_dir / 'best_single_model.pt'
                if best_model.exists():
                    checkpoint = best_model
                    print(f"✓ Found checkpoint: {checkpoint}")
                    break
                
                # Look for ensemble models
                ensemble_models = sorted(model_dir.glob('best_ensemble_model_*.pt'))
                if ensemble_models:
                    checkpoint = ensemble_models[0]
                    print(f"✓ Found checkpoint: {checkpoint}")
                    break
        
        if not checkpoint:
            print("❌ No checkpoint found!")
            print("\nPlease run with explicit checkpoint path:")
            print("   python test_final_model.py --checkpoint path/to/checkpoint.pt")
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
                print(f"✓ Found test features: {fdir / 'test_features.h5'}")
                break
            elif (fdir / 'test_features_multiscale.h5').exists():
                features_dir = fdir
                feature_suffix = '_multiscale'
                print(f"✓ Found test features: {fdir / 'test_features_multiscale.h5'}")
                break
        
        if not features_dir:
            print("❌ No test features found!")
            print("\nPlease run Stage 1 (feature extraction) first")
            sys.exit(1)
        
        # Run testing
        print(f"\n{'='*80}")
        print("STARTING AUTO TEST")
        print(f"{'='*80}\n")
        
        tester = FinalModelTester(
            checkpoint_path=checkpoint,
            features_dir=features_dir,
            output_dir='video_classification_project/test_results',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        try:
            standard_results, tta_results = tester.run_complete_test(
                batch_size=32,
                feature_suffix=feature_suffix
            )
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES:

1. AUTO MODE (searches for checkpoint and features automatically):
   $ python test_final_model.py

2. SINGLE MODEL TESTING:
   $ python test_final_model.py \
       --checkpoint video_classification_project/models_enhanced/best_single_model.pt \
       --features_dir video_classification_project/features_enhanced \
       --output_dir video_classification_project/test_results

3. WITH MULTISCALE FEATURES:
   $ python test_final_model.py \
       --checkpoint models_enhanced/best_single_model.pt \
       --features_dir features_enhanced \
       --feature_suffix _multiscale

4. ENSEMBLE TESTING (3 models):
   $ python test_final_model.py \
       --ensemble \
           models_enhanced/best_ensemble_model_0.pt \
           models_enhanced/best_ensemble_model_1.pt \
           models_enhanced/best_ensemble_model_2.pt \
       --features_dir features_enhanced \
       --output_dir test_results_ensemble

5. CUSTOM BATCH SIZE:
   $ python test_final_model.py \
       --checkpoint best_model.pt \
       --batch_size 64

OUTPUT FILES:
=============
Single Model:
- test_report.txt                      # Detailed text report
- test_results.json                    # JSON format results
- confusion_matrix_standard.png        # Standard testing CM
- confusion_matrix_tta.png             # TTA testing CM
- metrics_comparison.png               # Standard vs TTA comparison
- error_analysis_standard.png          # Error analysis (standard)
- error_analysis_tta.png               # Error analysis (TTA)

Ensemble:
- ensemble_test_report.txt
- ensemble_test_results.json
- ensemble_confusion_matrix_standard.png
- ensemble_confusion_matrix_tta.png

EXPECTED OUTPUT:
================

FINAL MODEL TESTING ON TEST SET
================================================================================
Checkpoint: best_single_model.pt
Features Dir: features_enhanced
Output Dir: test_results
Device: cuda
================================================================================

LOADING MODEL AND TEST DATA
======================================================================

Model Configuration:
   Feature dim: 2048
   Hidden dim: 768
   LSTM layers: 4
   Attention heads: 12
   Bidirectional: True
   Dropout: 0.4

Test Dataset:
   Samples: 150
   Classes: 4
   Categories: ['class1', 'class2', 'class3', 'class4']

✅ Model and data loaded successfully!

PHASE 1: STANDARD TESTING
================================================================================

Running inference on 150 samples...
Testing: 100%|██████████████| 5/5 [00:03<00:00]

======================================================================
STANDARD TEST RESULTS
======================================================================
Accuracy: 94.67%
Macro F1: 94.52%

Per-Class Metrics:
   class1:
      Precision: 96.00%
      Recall:    95.00%
      F1-Score:  95.50%
   class2:
      Precision: 94.00%
      Recall:    93.00%
      F1-Score:  93.50%
   ...

PHASE 2: TTA TESTING
================================================================================

TTA Mode: original
TTA original: 100%|████████| 5/5 [00:03<00:00]

TTA Mode: reverse
TTA reverse: 100%|█████████| 5/5 [00:03<00:00]

TTA Mode: speed_up
TTA speed_up: 100%|████████| 5/5 [00:03<00:00]

TTA Mode: speed_down
TTA speed_down: 100%|██████| 5/5 [00:03<00:00]

======================================================================
TTA TEST RESULTS
======================================================================
Accuracy: 96.00%
Macro F1: 95.88%

Per-Class Metrics:
   class1:
      Precision: 97.00%
      Recall:    96.00%
      F1-Score:  96.50%
   ...

GENERATING VISUALIZATIONS
================================================================================

   Saved: confusion_matrix_standard.png
   Saved: confusion_matrix_tta.png
   Saved: metrics_comparison.png

ERROR ANALYSIS - Standard
======================================================================
Total errors: 8 / 150 (5.33%)

Most Common Misclassifications:
   class1 → class2: 3 times
   class3 → class4: 2 times
   ...

   Saved: error_analysis_standard.png
   Saved: error_analysis_tta.png

================================================================================
FINAL MODEL TEST REPORT
================================================================================
Generated: 2025-01-15 14:30:00
Checkpoint: best_single_model.pt
Test samples: 150

STANDARD TESTING
--------------------------------------------------------------------------------
Accuracy: 94.67%
Macro F1: 94.52%

TTA TESTING
--------------------------------------------------------------------------------
Accuracy: 96.00%
Macro F1: 95.88%

TTA IMPROVEMENT
--------------------------------------------------------------------------------
Accuracy improvement: +1.33%
Macro F1 improvement: +1.36%

================================================================================

✅ TESTING COMPLETE!
================================================================================

Final Results:
   Standard Accuracy: 94.67%
   TTA Accuracy:      96.00%
   Improvement:       +1.33%

All results saved to: test_results

Generated files:
   - test_report.txt
   - test_results.json
   - confusion_matrix_standard.png
   - confusion_matrix_tta.png
   - metrics_comparison.png
   - error_analysis_standard.png
   - error_analysis_tta.png

NOTES:
======
- This script tests the final trained model on the held-out test set
- Uses both standard testing and Test-Time Augmentation (TTA)
- TTA typically improves accuracy by 1-2%
- Generates comprehensive visualizations and reports
- Supports both single model and ensemble testing
- Auto mode automatically finds checkpoints and features
"""