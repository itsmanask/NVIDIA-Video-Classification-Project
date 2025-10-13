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
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from collections import defaultdict
import pandas as pd

warnings.filterwarnings('ignore')

# Import from your training script
from model_train import (
    EnhancedCNNLSTM,
    MemoryEfficientVideoDataset,
    CheckpointManager
)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, checkpoint_path, data_dir, output_dir, device='cuda', split='val'):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.split = split  # 'val' or 'test'
        
        print(f"{'='*80}")
        print(f"MODEL EVALUATION & VISUALIZATION")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Split: {split}")
        print(f"Device: {self.device}")
        
    def load_model_and_data(self):
        """Load trained model and test dataset"""
        print(f"\n{'='*60}")
        print(f"LOADING MODEL AND DATA")
        print(f"{'='*60}")
        
        # Load checkpoint
        print(f"Loading checkpoint...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create test dataset
        print(f"\nLoading {self.split} dataset...")
        dataset = MemoryEfficientVideoDataset(
            self.data_dir,
            split=self.split,
            max_memory_gb=3.0,
            cache_size=20
        )
        
        if len(dataset) == 0:
            raise ValueError(f"{self.split} dataset is empty!")
        
        # Get category mapping
        self.category_mapping = dataset.category_mapping
        self.num_classes = len(self.category_mapping) if self.category_mapping else 4
        self.class_names = [None] * self.num_classes
        
        for name, idx in self.category_mapping.items():
            self.class_names[idx] = name
        
        print(f"\nCategories ({self.num_classes}):")
        for idx, name in enumerate(self.class_names):
            print(f"   {idx}: {name}")
        
        # Initialize model
        print(f"\nInitializing model...")
        model = EnhancedCNNLSTM(
            num_classes=self.num_classes,
            hidden_dim=512,
            num_layers=3,
            dropout=0.25,
            backbone='resnet50',
            bidirectional=True,
            attention=True
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Epoch trained: {checkpoint.get('epoch', 'unknown')}")
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"   Best Val Acc: {metrics.get('best_val_acc', 'N/A'):.2f}%")
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n{self.split.capitalize()} dataset: {len(dataset):,} samples ({len(loader)} batches)")
        
        return model, loader
    
    def get_predictions(self, model, loader):
        """Get all predictions and ground truth labels"""
        print(f"\n{'='*60}")
        print(f"GENERATING PREDICTIONS")
        print(f"{'='*60}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        model.eval()
        
        with torch.no_grad():
            with tqdm(total=len(loader), desc="Predicting") as pbar:
                for videos, labels in loader:
                    videos = videos.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(videos)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    pbar.update(1)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print(f"Total predictions: {len(all_preds):,}")
        
        return all_preds, all_labels, all_probs
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title_suffix=""):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix{title_suffix}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix{title_suffix}'
        
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
        
        filename = f"confusion_matrix{'_normalized' if normalize else ''}{title_suffix.lower().replace(' ', '_')}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_per_category_metrics(self, y_true, y_pred):
        """Plot detailed per-category metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Category': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Sort by F1-score
        metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
        
        # Plot
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
        
        # Support (sample count)
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
        
        save_path = self.output_dir / 'per_category_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: per_category_metrics.png")
        plt.close()
        
        return metrics_df
    
    def plot_roc_curves(self, y_true, y_probs):
        """Plot ROC curves for each category"""
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
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
        
        save_path = self.output_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: roc_curves.png")
        plt.close()
        
        return roc_auc
    
    def plot_prediction_distribution(self, y_true, y_pred):
        """Plot prediction distribution and error analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # True vs Predicted distribution
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
        
        # Error distribution
        errors = (y_true != y_pred).astype(int)
        error_by_class = np.array([
            errors[y_true == i].sum() for i in range(self.num_classes)
        ])
        total_by_class = np.array([
            (y_true == i).sum() for i in range(self.num_classes)
        ])
        error_rate = error_by_class / (total_by_class + 1e-10) * 100
        
        bars = axes[1].bar(self.class_names, error_rate, color='crimson', alpha=0.7)
        axes[1].set_xlabel('Category', fontweight='bold')
        axes[1].set_ylabel('Error Rate (%)', fontweight='bold')
        axes[1].set_title('Error Rate per Category', fontweight='bold', fontsize=14)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'prediction_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: prediction_distribution.png")
        plt.close()
    
    def plot_confidence_analysis(self, y_true, y_pred, y_probs):
        """Analyze prediction confidence"""
        # Get confidence (max probability) for each prediction
        confidence = np.max(y_probs, axis=1)
        correct = (y_true == y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confidence distribution for correct vs incorrect
        axes[0, 0].hist(confidence[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0, 0].hist(confidence[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence', fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontweight='bold')
        axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy vs Confidence bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(bins)-1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
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
                           f'{height:.3f}',
                           ha='center', va='bottom', fontweight='bold')
        
        # Top-2 confidence gap (uncertainty measure)
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
        
        save_path = self.output_dir / 'confidence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: confidence_analysis.png")
        plt.close()
    
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
        
        # Best and worst performing categories
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
        
        # Find top misclassifications
        misclass = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    misclass.append((cm[i, j], self.class_names[i], self.class_names[j]))
        
        misclass.sort(reverse=True)
        for count, true_class, pred_class in misclass[:5]:
            report.append(f"{true_class} -> {pred_class}: {count} samples")  # Use -> instead of Unicode arrow
        
        report.append("")
        report.append("="*80)
        
        # Save report with UTF-8 encoding
        report_text = "\n".join(report)
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nSaved: evaluation_report.txt")
        print(f"\n{report_text}")
        
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
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print(f"\n{'='*80}")
        print(f"STARTING FULL EVALUATION PIPELINE")
        print(f"{'='*80}\n")
        
        # Load model and data
        model, test_loader = self.load_model_and_data()
        
        # Get predictions
        y_pred, y_true, y_probs = self.get_predictions(model, test_loader)
        
        # Generate all visualizations
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")
        
        # 1. Confusion matrices
        print("1. Confusion matrices...")
        self.plot_confusion_matrix(y_true, y_pred, normalize=False)
        self.plot_confusion_matrix(y_true, y_pred, normalize=True)
        
        # 2. Per-category metrics
        print("\n2. Per-category metrics...")
        metrics_df = self.plot_per_category_metrics(y_true, y_pred)
        
        # 3. ROC curves
        print("\n3. ROC curves...")
        roc_auc = self.plot_roc_curves(y_true, y_probs)
        
        # 4. Prediction distribution
        print("\n4. Prediction distribution...")
        self.plot_prediction_distribution(y_true, y_pred)
        
        # 5. Confidence analysis
        print("\n5. Confidence analysis...")
        self.plot_confidence_analysis(y_true, y_pred, y_probs)
        
        # Generate reports
        print(f"\n{'='*60}")
        print(f"GENERATING REPORTS")
        print(f"{'='*60}\n")
        
        self.generate_detailed_report(y_true, y_pred, y_probs, metrics_df, roc_auc)
        self.save_results_json(y_true, y_pred, y_probs, metrics_df, roc_auc)
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"   • confusion_matrix.png")
        print(f"   • confusion_matrix_normalized.png")
        print(f"   • per_category_metrics.png")
        print(f"   • roc_curves.png")
        print(f"   • prediction_distribution.png")
        print(f"   • confidence_analysis.png")
        print(f"   • evaluation_report.txt")
        print(f"   • evaluation_results.json")


def main():
    """Main function to run evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained video classification model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/best_model.pt)'
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
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset split to evaluate (default: val)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        
        # Try to find checkpoints
        possible_dirs = [
            Path('video_classification_project/models/checkpoints'),
            Path('models/checkpoints'),
            Path('checkpoints')
        ]
        
        found_checkpoints = False
        for checkpoint_dir in possible_dirs:
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('*.pt'))
                if checkpoints:
                    found_checkpoints = True
                    print(f"\nIn {checkpoint_dir}:")
                    for ckpt in sorted(checkpoints):
                        print(f"   {ckpt}")
        
        if not found_checkpoints:
            print("   No checkpoints found")
        
        return
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("\nPlease specify the correct path to your processed data directory")
        return
    
    # Create evaluator and run
    evaluator = ModelEvaluator(
        checkpoint_path=checkpoint_path,
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
    # Example usage if run without arguments
    import sys
    
    if len(sys.argv) == 1:
        print("="*80)
        print("MODEL EVALUATION SCRIPT - AUTO MODE")
        print("="*80)
        print("\nSearching for best_model.pt...")
        
        # Auto-detect best_model.pt
        possible_paths = [
            Path('video_classification_project/models/checkpoints/best_model.pt'),
            Path('models/checkpoints/best_model.pt'),
            Path('checkpoints/best_model.pt'),
            Path('best_model.pt')
        ]
        
        best_model_path = None
        for path in possible_paths:
            if path.exists():
                best_model_path = path
                print(f"✓ Found best_model.pt at: {path}")
                break
        
        if best_model_path is None:
            print("\n❌ best_model.pt not found in standard locations.")
            print("\nSearched in:")
            for path in possible_paths:
                print(f"   {path}")
            
            print("\n" + "="*80)
            print("Searching for other checkpoints...")
            
            # Show available checkpoints
            possible_dirs = [
                Path('video_classification_project/models/checkpoints'),
                Path('models/checkpoints'),
                Path('checkpoints')
            ]
            
            found_any = False
            for checkpoint_dir in possible_dirs:
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob('*.pt'))
                    if checkpoints:
                        if not found_any:
                            print("\nFound checkpoints:")
                            found_any = True
                        print(f"\nIn {checkpoint_dir}:")
                        for ckpt in sorted(checkpoints)[:10]:
                            print(f"   {ckpt}")
                        if len(checkpoints) > 10:
                            print(f"   ... and {len(checkpoints)-10} more")
            
            if not found_any:
                print("\nNo checkpoints found.")
            
            print("\n" + "="*80)
            print("\nUsage:")
            print("  python model_evaluation.py --checkpoint <path_to_checkpoint>")
            print("\nExample:")
            print("  python model_evaluation.py --checkpoint models/checkpoints/checkpoint_epoch_045.pt")
            print("\nOptional arguments:")
            print("  --data_dir     Path to processed data (default: video_classification_project/data/processed)")
            print("  --output_dir   Output directory for results (default: evaluation_results)")
            print("  --device       Device to use: cuda or cpu (default: cuda)")
            sys.exit(1)
        
        # Find data directory
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
                print(f"✓ Found data directory at: {path}")
                break
        
        if data_dir is None:
            print("\n❌ Data directory not found.")
            print("Please specify with --data_dir argument")
            sys.exit(1)
        
        # Run evaluation with auto-detected paths
        print("\n" + "="*80)
        print("STARTING AUTO EVALUATION")
        print("="*80)
        print(f"Checkpoint: {best_model_path}")
        print(f"Data Dir:   {data_dir}")
        print(f"Output Dir: evaluation_results")
        print(f"Split:      val")
        print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("="*80 + "\n")
        
        evaluator = ModelEvaluator(
            checkpoint_path=best_model_path,
            data_dir=data_dir,
            output_dir='evaluation_results',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            split='val'
        )
        
        try:
            evaluator.run_full_evaluation()
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        main()