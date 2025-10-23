import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class EnhancedCheckpointAnalyzer:
    """Comprehensive analyzer for PyTorch checkpoint files with advanced visualizations"""
    
    def __init__(self):
        self.checkpoint_data = {}
        self.checkpoint_path = None
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load and analyze a checkpoint file"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path.name}")
        print(f"File size: {checkpoint_path.stat().st_size / 1e6:.2f}MB")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.checkpoint_data = checkpoint
            self.checkpoint_path = checkpoint_path
            
            print(f"✓ Checkpoint loaded successfully")
            return checkpoint
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return {}
    
    def analyze_checkpoint(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze checkpoint contents and extract key information"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint:
            print("No checkpoint data available")
            return {}
        
        analysis = {
            'basic_info': {},
            'training_progress': {},
            'model_info': {},
            'optimizer_info': {},
            'metrics': {},
            'file_info': {}
        }
        
        print(f"\n{'='*70}")
        print(f"CHECKPOINT ANALYSIS")
        print(f"{'='*70}")
        
        # Basic Information
        print(f"\n📋 BASIC INFORMATION")
        print(f"{'-'*50}")
        
        basic_info = {}
        
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            basic_info['epoch'] = epoch
            print(f"   Epoch: {epoch}")
        
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
            basic_info['best_epoch'] = best_epoch
            print(f"   Best Epoch: {best_epoch}")
        
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc']
            basic_info['best_val_acc'] = best_val_acc
            print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        
        if 'patience_counter' in checkpoint:
            patience_counter = checkpoint['patience_counter']
            basic_info['patience_counter'] = patience_counter
            print(f"   Patience Counter: {patience_counter}")
        
        analysis['basic_info'] = basic_info
        
        # Model Information
        print(f"\n🧠 MODEL INFORMATION")
        print(f"{'-'*50}")
        
        model_info = {}
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Count parameters
            total_params = 0
            trainable_params = 0
            layer_info = []
            
            for name, param in state_dict.items():
                param_count = param.numel()
                total_params += param_count
                param_size_mb = param.numel() * 4 / 1e6
                
                layer_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'params': param_count,
                    'size_mb': param_size_mb
                })
            
            model_info['total_parameters'] = total_params
            model_info['total_size_mb'] = total_params * 4 / 1e6
            model_info['layer_count'] = len(state_dict)
            model_info['layer_info'] = layer_info
            
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Model Size: {total_params * 4 / 1e6:.2f}MB")
            print(f"   Number of Layers: {len(state_dict)}")
            
            # Analyze architecture
            architecture_info = self._analyze_architecture(state_dict)
            model_info.update(architecture_info)
            
            if architecture_info['has_lstm']:
                print(f"   ✓ Contains LSTM layers")
            if architecture_info['has_attention']:
                print(f"   ✓ Contains Attention layers")
            if architecture_info['has_classifier']:
                print(f"   ✓ Contains Classifier")
        
        analysis['model_info'] = model_info
        
        # Optimizer Information
        print(f"\n⚙️ OPTIMIZER INFORMATION")
        print(f"{'-'*50}")
        
        optimizer_info = {}
        
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            
            if 'param_groups' in opt_state:
                param_groups = opt_state['param_groups']
                if param_groups:
                    group = param_groups[0]
                    
                    for key, value in group.items():
                        if key != 'params':
                            optimizer_info[key] = value
                            if key == 'lr':
                                print(f"   Learning Rate: {value:.6f}")
                            elif key == 'weight_decay':
                                print(f"   Weight Decay: {value}")
                            elif key == 'betas':
                                print(f"   Betas: {value}")
        
        analysis['optimizer_info'] = optimizer_info
        
        # Training Metrics
        print(f"\n📊 TRAINING METRICS")
        print(f"{'-'*50}")
        
        metrics = {}
        
        if 'metrics' in checkpoint:
            checkpoint_metrics = checkpoint['metrics']
            metrics['current'] = checkpoint_metrics
            
            for key in ['accuracy', 'f1_weighted', 'f1_per_class']:
                if key in checkpoint_metrics:
                    value = checkpoint_metrics[key]
                    if key == 'f1_per_class':
                        print(f"   Per-class F1: {[f'{f:.2f}' for f in value]}")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
        
        # Training History
        if 'history' in checkpoint:
            history = checkpoint['history']
            metrics['history'] = history
            
            print(f"\n   📈 Training History Summary:")
            for key in ['train_acc', 'val_acc', 'train_loss', 'val_loss']:
                if key in history and history[key]:
                    values = history[key]
                    latest = values[-1]
                    best_idx = np.argmax(values) if 'acc' in key else np.argmin(values)
                    best = values[best_idx]
                    
                    if 'acc' in key:
                        print(f"      {key}: Latest={latest:.2f}%, Best={best:.2f}% (epoch {best_idx})")
                    else:
                        print(f"      {key}: Latest={latest:.4f}, Best={best:.4f} (epoch {best_idx})")
        
        analysis['metrics'] = metrics
        
        return analysis
    
    def _analyze_architecture(self, state_dict: Dict) -> Dict[str, Any]:
        """Analyze model architecture from state dict"""
        architecture = {
            'has_cnn': False,
            'has_lstm': False,
            'has_attention': False,
            'has_classifier': False,
            'layer_types': {}
        }
        
        for name in state_dict.keys():
            if 'conv' in name.lower():
                architecture['has_cnn'] = True
                architecture['layer_types']['conv'] = architecture['layer_types'].get('conv', 0) + 1
            elif 'lstm' in name.lower():
                architecture['has_lstm'] = True
                architecture['layer_types']['lstm'] = architecture['layer_types'].get('lstm', 0) + 1
            elif 'attention' in name.lower() or 'attn' in name.lower():
                architecture['has_attention'] = True
                architecture['layer_types']['attention'] = architecture['layer_types'].get('attention', 0) + 1
            elif 'classifier' in name.lower() or 'fc' in name.lower():
                architecture['has_classifier'] = True
                architecture['layer_types']['classifier'] = architecture['layer_types'].get('classifier', 0) + 1
            elif 'norm' in name.lower() or 'bn' in name.lower():
                architecture['layer_types']['normalization'] = architecture['layer_types'].get('normalization', 0) + 1
        
        return architecture
    
    def plot_comprehensive_dashboard(self, checkpoint: Optional[Dict] = None, 
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (24, 20)):
        """Create comprehensive visualization dashboard"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint or 'history' not in checkpoint:
            print("No training history available for plotting")
            return
        
        history = checkpoint['history']
        current_epoch = checkpoint.get('epoch', len(history.get('train_loss', [])) - 1)
        
        print(f"\n🎨 Creating comprehensive visualization dashboard...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # Title
        fig.suptitle(f'Training Analysis Dashboard - Epoch {current_epoch}',
                    fontsize=22, fontweight='bold', y=0.98)
        
        # 1. Training & Validation Loss
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1, history, current_epoch)
        
        # 2. Training & Validation Accuracy
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_accuracy_curves(ax2, history, current_epoch)
        
        # 3. Overfitting Analysis (Train-Val Gap)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_overfitting_analysis(ax3, history)
        
        # 4. Learning Rate Schedule
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_learning_rate(ax4, history)
        
        # 5. Per-class F1 Score Progression
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_per_class_f1(ax5, history)
        
        # 6. Loss Distribution (Box plot of recent epochs)
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_loss_distribution(ax6, history)
        
        # 7. Accuracy Improvement Rate
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_improvement_rate(ax7, history)
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_summary_stats(ax8, checkpoint)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Dashboard saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_loss_curves(self, ax, history, current_epoch):
        """Plot training and validation loss"""
        if 'train_loss' in history and history['train_loss']:
            epochs = range(len(history['train_loss']))
            ax.plot(epochs, history['train_loss'], 'b-', linewidth=2.5, alpha=0.8, label='Train Loss')
            
        if 'val_loss' in history and history['val_loss']:
            epochs = range(len(history['val_loss']))
            ax.plot(epochs, history['val_loss'], 'r--', linewidth=2.5, alpha=0.8, label='Val Loss')
            
            # Mark best epoch
            best_epoch = np.argmin(history['val_loss'])
            best_loss = history['val_loss'][best_epoch]
            ax.scatter([best_epoch], [best_loss], color='gold', s=200, zorder=5, 
                      marker='*', edgecolors='black', linewidth=2)
            ax.annotate(f'Best\n{best_loss:.4f}', xy=(best_epoch, best_loss),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
    
    def _plot_accuracy_curves(self, ax, history, current_epoch):
        """Plot training and validation accuracy"""
        if 'train_acc' in history and history['train_acc']:
            epochs = range(len(history['train_acc']))
            ax.plot(epochs, history['train_acc'], 'b-', linewidth=2.5, alpha=0.8, label='Train Acc')
            
        if 'val_acc' in history and history['val_acc']:
            epochs = range(len(history['val_acc']))
            ax.plot(epochs, history['val_acc'], 'g--', linewidth=2.5, alpha=0.8, label='Val Acc')
            
            # Mark best epoch
            best_epoch = np.argmax(history['val_acc'])
            best_acc = history['val_acc'][best_epoch]
            ax.scatter([best_epoch], [best_acc], color='gold', s=200, zorder=5,
                      marker='*', edgecolors='black', linewidth=2)
            ax.annotate(f'Best\n{best_acc:.2f}%', xy=(best_epoch, best_acc),
                       xytext=(10, 15), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
    
    def _plot_overfitting_analysis(self, ax, history):
        """Plot train-val gap to analyze overfitting"""
        if 'train_acc' in history and 'val_acc' in history:
            train_acc = history['train_acc']
            val_acc = history['val_acc']
            min_len = min(len(train_acc), len(val_acc))
            
            gap = [train_acc[i] - val_acc[i] for i in range(min_len)]
            epochs = range(min_len)
            
            colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gap]
            ax.bar(epochs, gap, color=colors, alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
            ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
            
            ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
            ax.set_ylabel('Train-Val Gap (%)', fontsize=10, fontweight='bold')
            ax.set_title('Overfitting Analysis', fontsize=12, fontweight='bold', pad=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_learning_rate(self, ax, history):
        """Plot learning rate schedule"""
        if 'learning_rates' in history and history['learning_rates']:
            epochs = range(len(history['learning_rates']))
            lr_values = history['learning_rates']
            
            ax.plot(epochs, lr_values, 'purple', linewidth=2.5, marker='o', markersize=3)
            ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=10, fontweight='bold')
            ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold', pad=10)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_facecolor('#f9f9f9')
        else:
            ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    def _plot_per_class_f1(self, ax, history):
        """Plot per-class F1 score progression"""
        if 'val_per_class_f1' in history and history['val_per_class_f1']:
            per_class_f1 = np.array(history['val_per_class_f1'])
            num_classes = per_class_f1.shape[1]
            epochs = range(len(per_class_f1))
            
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            for class_idx in range(num_classes):
                class_f1 = per_class_f1[:, class_idx]
                ax.plot(epochs, class_f1, color=colors[class_idx], 
                       linewidth=2, alpha=0.8, label=f'Class {class_idx}',
                       marker='o' if len(epochs) < 30 else None, markersize=3)
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
            ax.set_title('Per-Class F1 Score Progression', fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10, ncol=num_classes if num_classes <= 4 else 2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f9f9f9')
        else:
            ax.text(0.5, 0.5, 'No per-class F1 data', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    def _plot_loss_distribution(self, ax, history):
        """Plot distribution of recent losses"""
        if 'val_loss' in history and len(history['val_loss']) >= 10:
            recent_losses = history['val_loss'][-20:]
            
            ax.boxplot([recent_losses], vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
            
            ax.set_ylabel('Validation Loss', fontsize=10, fontweight='bold')
            ax.set_title('Recent Loss Distribution', fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks([1])
            ax.set_xticklabels(['Last 20 Epochs'])
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    def _plot_improvement_rate(self, ax, history):
        """Plot rate of improvement in validation accuracy"""
        if 'val_acc' in history and len(history['val_acc']) > 5:
            val_acc = np.array(history['val_acc'])
            # Calculate moving average improvement
            window = min(5, len(val_acc) // 4)
            improvement = np.diff(val_acc)
            
            if len(improvement) > window:
                smoothed = pd.Series(improvement).rolling(window=window, center=True).mean()
                epochs = range(1, len(val_acc))
                
                colors = ['green' if x > 0 else 'red' for x in smoothed]
                ax.bar(epochs, smoothed, color=colors, alpha=0.6, width=0.8)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                
                ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
                ax.set_ylabel('Acc Change (%)', fontsize=10, fontweight='bold')
                ax.set_title('Accuracy Improvement Rate', fontsize=12, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    def _plot_summary_stats(self, ax, checkpoint):
        """Display summary statistics"""
        ax.axis('off')
        
        stats_text = "📊 Summary Statistics\n" + "="*30 + "\n\n"
        
        if 'epoch' in checkpoint:
            stats_text += f"Current Epoch: {checkpoint['epoch']}\n"
        
        if 'best_epoch' in checkpoint:
            stats_text += f"Best Epoch: {checkpoint['best_epoch']}\n"
        
        if 'best_val_acc' in checkpoint:
            stats_text += f"Best Val Acc: {checkpoint['best_val_acc']:.2f}%\n"
        
        if 'patience_counter' in checkpoint:
            stats_text += f"Patience: {checkpoint['patience_counter']}/25\n"
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            
            if 'val_acc' in history and history['val_acc']:
                recent_acc = history['val_acc'][-5:]
                stats_text += f"\nRecent Avg Acc: {np.mean(recent_acc):.2f}%\n"
                stats_text += f"Acc Std Dev: {np.std(recent_acc):.2f}%\n"
            
            if 'val_loss' in history and history['val_loss']:
                recent_loss = history['val_loss'][-5:]
                stats_text += f"\nRecent Avg Loss: {np.mean(recent_loss):.4f}\n"
                stats_text += f"Loss Std Dev: {np.std(recent_loss):.4f}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None,
                             save_path: Optional[str] = None, figsize=(10, 8)):
        """Plot confusion matrix with detailed metrics"""
        print(f"\n🎯 Creating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    def plot_model_architecture(self, checkpoint: Optional[Dict] = None,
                                save_path: Optional[str] = None, figsize=(12, 8)):
        """Visualize model architecture breakdown"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if 'model_state_dict' not in checkpoint:
            print("No model state dict available")
            return
        
        print(f"\n🏗️ Creating model architecture visualization...")
        
        state_dict = checkpoint['model_state_dict']
        
        # Analyze layers
        layer_types = {}
        layer_sizes = {}
        
        for name, param in state_dict.items():
            # Extract layer type
            if 'weight' in name:
                layer_name = name.split('.weight')[0]
                
                # Categorize layer
                if 'lstm' in layer_name.lower():
                    layer_type = 'LSTM'
                elif 'attention' in layer_name.lower() or 'attn' in layer_name.lower():
                    layer_type = 'Attention'
                elif 'classifier' in layer_name.lower() or 'fc' in layer_name.lower():
                    layer_type = 'Classifier'
                elif 'norm' in layer_name.lower():
                    layer_type = 'Normalization'
                elif 'conv' in layer_name.lower():
                    layer_type = 'Convolution'
                elif 'linear' in layer_name.lower():
                    layer_type = 'Linear'
                else:
                    layer_type = 'Other'
                
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
                layer_sizes[layer_type] = layer_sizes.get(layer_type, 0) + param.numel()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Layer count pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
        wedges, texts, autotexts = ax1.pie(layer_types.values(), labels=layer_types.keys(),
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Layer Type Distribution', fontsize=14, fontweight='bold', pad=15)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Parameter size bar chart
        types = list(layer_sizes.keys())
        sizes = [layer_sizes[t] / 1e6 for t in types]  # Convert to millions
        
        bars = ax2.barh(types, sizes, color=colors[:len(types)])
        ax2.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax2.set_title('Parameter Count by Layer Type', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}M', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Architecture visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_stability(self, checkpoint: Optional[Dict] = None,
                                save_path: Optional[str] = None, figsize=(15, 10)):
        """Analyze training stability with various metrics"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if 'history' not in checkpoint:
            print("No training history available")
            return
        
        print(f"\n📈 Creating training stability analysis...")
        
        history = checkpoint['history']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Stability Analysis', fontsize=18, fontweight='bold')
        
        # 1. Loss variance over time
        ax = axes[0, 0]
        if 'val_loss' in history and len(history['val_loss']) > 10:
            window = 5
            val_loss = np.array(history['val_loss'])
            rolling_std = pd.Series(val_loss).rolling(window=window).std()
            epochs = range(len(rolling_std))
            
            ax.plot(epochs, rolling_std, 'b-', linewidth=2.5)
            ax.fill_between(epochs, 0, rolling_std, alpha=0.3)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss Std Dev', fontsize=11, fontweight='bold')
            ax.set_title(f'Loss Stability (Rolling Std, window={window})', 
                        fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
        
        # 2. Gradient of validation accuracy
        ax = axes[0, 1]
        if 'val_acc' in history and len(history['val_acc']) > 5:
            val_acc = np.array(history['val_acc'])
            gradient = np.gradient(val_acc)
            epochs = range(len(gradient))
            
            colors = ['green' if g > 0 else 'red' for g in gradient]
            ax.bar(epochs, gradient, color=colors, alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy Change', fontsize=11, fontweight='bold')
            ax.set_title('Validation Accuracy Gradient', fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Training convergence (moving average)
        ax = axes[1, 0]
        if 'train_loss' in history and len(history['train_loss']) > 10:
            train_loss = np.array(history['train_loss'])
            window = min(10, len(train_loss) // 3)
            moving_avg = pd.Series(train_loss).rolling(window=window, center=True).mean()
            epochs = range(len(train_loss))
            
            ax.plot(epochs, train_loss, 'lightblue', alpha=0.5, label='Raw')
            ax.plot(epochs, moving_avg, 'darkblue', linewidth=2.5, label=f'MA (window={window})')
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
            ax.set_title('Training Convergence', fontsize=12, fontweight='bold', pad=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 4. Epoch-to-epoch improvement consistency
        ax = axes[1, 1]
        if 'val_acc' in history and len(history['val_acc']) > 10:
            val_acc = np.array(history['val_acc'])
            improvements = np.diff(val_acc)
            
            # Count positive vs negative changes
            positive = np.sum(improvements > 0)
            negative = np.sum(improvements <= 0)
            
            ax.bar(['Improving', 'Not Improving'], [positive, negative], 
                  color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Number of Epochs', fontsize=11, fontweight='bold')
            ax.set_title('Improvement Consistency', fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentages
            total = positive + negative
            for i, (val, label) in enumerate([(positive, 'Improving'), (negative, 'Not Improving')]):
                pct = 100 * val / total
                ax.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Stability analysis saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_performance(self, checkpoint: Optional[Dict] = None,
                                   save_path: Optional[str] = None, 
                                   class_names=None, figsize=(14, 10)):
        """Detailed per-class performance analysis"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if 'history' not in checkpoint or 'val_per_class_f1' not in checkpoint['history']:
            print("No per-class F1 data available")
            return
        
        print(f"\n🎯 Creating per-class performance analysis...")
        
        history = checkpoint['history']
        per_class_f1 = np.array(history['val_per_class_f1'])
        num_classes = per_class_f1.shape[1]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Per-Class Performance Analysis', fontsize=18, fontweight='bold')
        
        # 1. F1 progression for each class
        ax1 = fig.add_subplot(gs[0, :])
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        epochs = range(len(per_class_f1))
        
        for class_idx in range(num_classes):
            class_f1 = per_class_f1[:, class_idx]
            ax1.plot(epochs, class_f1, color=colors[class_idx], 
                    linewidth=2.5, alpha=0.8, label=class_names[class_idx],
                    marker='o' if len(epochs) < 30 else None, markersize=4)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('F1 Score Progression by Class', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=10, ncol=2 if num_classes > 4 else 1)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f9f9f9')
        
        # 2. Current F1 scores bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        current_f1 = per_class_f1[-1]
        bars = ax2.bar(class_names, current_f1, color=colors, alpha=0.7)
        ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Current F1 Scores', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Best F1 scores achieved
        ax3 = fig.add_subplot(gs[1, 1])
        best_f1 = np.max(per_class_f1, axis=0)
        bars = ax3.bar(class_names, best_f1, color=colors, alpha=0.7)
        ax3.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Best F1 Scores Achieved', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. F1 improvement from start to current
        ax4 = fig.add_subplot(gs[2, 0])
        start_f1 = per_class_f1[0]
        improvement = current_f1 - start_f1
        colors_improvement = ['green' if imp > 0 else 'red' for imp in improvement]
        bars = ax4.bar(class_names, improvement, color=colors_improvement, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_ylabel('F1 Improvement (%)', fontsize=11, fontweight='bold')
        ax4.set_title('F1 Improvement (Start → Current)', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 5. Class F1 variance (stability)
        ax5 = fig.add_subplot(gs[2, 1])
        f1_std = np.std(per_class_f1, axis=0)
        bars = ax5.bar(class_names, f1_std, color=colors, alpha=0.7)
        ax5.set_ylabel('F1 Std Dev (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Training Stability by Class', fontsize=12, fontweight='bold', pad=10)
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Per-class analysis saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_checkpoints(self, checkpoint_paths: List[str]) -> pd.DataFrame:
        """Compare multiple checkpoints"""
        print(f"\n📊 Comparing {len(checkpoint_paths)} checkpoints...")
        
        comparisons = []
        
        for path in checkpoint_paths:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                
                comparison = {
                    'file': Path(path).name,
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'best_epoch': checkpoint.get('best_epoch', 'N/A'),
                    'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
                    'patience': checkpoint.get('patience_counter', 'N/A'),
                }
                
                # Get final metrics from history
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    if 'val_acc' in history and history['val_acc']:
                        comparison['final_val_acc'] = history['val_acc'][-1]
                    if 'val_loss' in history and history['val_loss']:
                        comparison['final_val_loss'] = history['val_loss'][-1]
                
                comparisons.append(comparison)
                print(f"✓ Analyzed {Path(path).name}")
                
            except Exception as e:
                print(f"✗ Failed to analyze {Path(path).name}: {e}")
                continue
        
        if comparisons:
            df = pd.DataFrame(comparisons)
            print(f"\n{'='*70}")
            print("CHECKPOINT COMPARISON")
            print(f"{'='*70}")
            print(df.to_string(index=False))
            return df
        
        return pd.DataFrame()
    
    def generate_comprehensive_report(self, checkpoint: Optional[Dict] = None,
                                     output_dir: Optional[str] = None):
        """Generate all visualizations and reports"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint:
            print("No checkpoint data available")
            return
        
        if output_dir is None:
            output_dir = Path('.')
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = self.checkpoint_path.stem if self.checkpoint_path else 'checkpoint'
        
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*70}")
        
        # 1. Comprehensive dashboard
        self.plot_comprehensive_dashboard(
            checkpoint,
            save_path=output_dir / f'{base_name}_dashboard.png'
        )
        
        # 2. Model architecture
        self.plot_model_architecture(
            checkpoint,
            save_path=output_dir / f'{base_name}_architecture.png'
        )
        
        # 3. Training stability
        self.plot_training_stability(
            checkpoint,
            save_path=output_dir / f'{base_name}_stability.png'
        )
        
        # 4. Per-class performance
        if 'history' in checkpoint and 'val_per_class_f1' in checkpoint['history']:
            self.plot_per_class_performance(
                checkpoint,
                save_path=output_dir / f'{base_name}_per_class.png'
            )
        
        # 5. Text report
        report = self._generate_text_report(checkpoint)
        report_path = output_dir / f'{base_name}_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Text report saved to {report_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ ALL REPORTS GENERATED SUCCESSFULLY")
        print(f"📁 Output directory: {output_dir}")
        print(f"{'='*70}")
    
    def _generate_text_report(self, checkpoint: Dict) -> str:
        """Generate detailed text report"""
        report = []
        report.append("="*70)
        report.append("COMPREHENSIVE CHECKPOINT ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.checkpoint_path:
            report.append(f"File: {self.checkpoint_path}")
            report.append(f"Size: {self.checkpoint_path.stat().st_size / 1e6:.2f}MB")
        
        # Basic info
        report.append(f"\n{'='*70}")
        report.append("BASIC INFORMATION")
        report.append(f"{'='*70}")
        
        if 'epoch' in checkpoint:
            report.append(f"Current Epoch: {checkpoint['epoch']}")
        if 'best_epoch' in checkpoint:
            report.append(f"Best Epoch: {checkpoint['best_epoch']}")
        if 'best_val_acc' in checkpoint:
            report.append(f"Best Validation Accuracy: {checkpoint['best_val_acc']:.2f}%")
        if 'patience_counter' in checkpoint:
            report.append(f"Early Stopping Patience: {checkpoint['patience_counter']}/25")
        
        # Model info
        if 'model_state_dict' in checkpoint:
            report.append(f"\n{'='*70}")
            report.append("MODEL ARCHITECTURE")
            report.append(f"{'='*70}")
            
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            
            report.append(f"Total Parameters: {total_params:,}")
            report.append(f"Model Size: {total_params * 4 / 1e6:.2f}MB")
            report.append(f"Number of Layers: {len(state_dict)}")
        
        # Training history summary
        if 'history' in checkpoint:
            report.append(f"\n{'='*70}")
            report.append("TRAINING HISTORY SUMMARY")
            report.append(f"{'='*70}")
            
            history = checkpoint['history']
            
            for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
                if metric in history and history[metric]:
                    values = history[metric]
                    
                    if 'acc' in metric:
                        best_val = max(values)
                        best_epoch = values.index(best_val)
                        current_val = values[-1]
                        report.append(f"\n{metric.replace('_', ' ').title()}:")
                        report.append(f"  Current: {current_val:.2f}%")
                        report.append(f"  Best: {best_val:.2f}% (epoch {best_epoch})")
                    else:
                        best_val = min(values)
                        best_epoch = values.index(best_val)
                        current_val = values[-1]
                        report.append(f"\n{metric.replace('_', ' ').title()}:")
                        report.append(f"  Current: {current_val:.4f}")
                        report.append(f"  Best: {best_val:.4f} (epoch {best_epoch})")
        
        # Per-class performance
        if 'history' in checkpoint and 'val_per_class_f1' in checkpoint['history']:
            report.append(f"\n{'='*70}")
            report.append("PER-CLASS PERFORMANCE")
            report.append(f"{'='*70}")
            
            per_class_f1 = np.array(checkpoint['history']['val_per_class_f1'])
            num_classes = per_class_f1.shape[1]
            
            current_f1 = per_class_f1[-1]
            best_f1 = np.max(per_class_f1, axis=0)
            
            for class_idx in range(num_classes):
                report.append(f"\nClass {class_idx}:")
                report.append(f"  Current F1: {current_f1[class_idx]:.2f}%")
                report.append(f"  Best F1: {best_f1[class_idx]:.2f}%")
        
        report.append(f"\n{'='*70}")
        report.append("END OF REPORT")
        report.append(f"{'='*70}")
        
        return "\n".join(report)


def main():
    """Main function with enhanced options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced PyTorch Checkpoint Analyzer with Visualizations')
    parser.add_argument('checkpoint', nargs='*', help='Path(s) to checkpoint file(s)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple checkpoints')
    parser.add_argument('--full-report', action='store_true', help='Generate all visualizations and reports')
    parser.add_argument('--dashboard', action='store_true', help='Generate comprehensive dashboard')
    parser.add_argument('--architecture', action='store_true', help='Visualize model architecture')
    parser.add_argument('--stability', action='store_true', help='Analyze training stability')
    parser.add_argument('--per-class', action='store_true', help='Per-class performance analysis')
    parser.add_argument('--output-dir', type=str, 
                       default='video_classification_project/results/checkpoint_analysis', 
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Set default output directory
    default_output = Path('video_classification_project/results/checkpoint_analysis')
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    
    if not args.checkpoint:
        # Interactive mode - search for checkpoints
        print("🔍 Searching for checkpoint files...")
        
        search_paths = [
            Path('.'),
            Path('checkpoints'),
            Path('models'),
            Path('models_enhanced'),
            Path('video_classification_project/models'),
            Path('video_classification_project/models/checkpoints'),
            Path('video_classification_project/models_enhanced')
        ]
        
        checkpoint_files = []
        seen_files = set()
        
        for search_path in search_paths:
            if search_path.exists():
                # Search for checkpoint files
                found_files = (list(search_path.glob('*.pt')) + 
                             list(search_path.glob('best_*.pt')) +
                             list(search_path.glob('*checkpoint*.pt')))
                for file in found_files:
                    resolved_path = file.resolve()
                    if resolved_path not in seen_files and 'checkpoint' in file.name.lower() or 'best' in file.name.lower() or 'model' in file.name.lower():
                        checkpoint_files.append(file)
                        seen_files.add(resolved_path)
        
        if not checkpoint_files:
            print("❌ No checkpoint files found!")
            print("\n📁 Searched in:")
            for path in search_paths:
                if path.exists():
                    print(f"   - {path.absolute()}")
            print("\nPlease specify checkpoint file(s) as arguments:")
            print("   python checkpoint_analyzer_enhanced.py path/to/checkpoint.pt")
            return
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"\n✓ Found {len(checkpoint_files)} checkpoint file(s):")
        for i, file in enumerate(checkpoint_files[:15]):  # Show first 15
            mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            size_mb = file.stat().st_size / 1e6
            print(f"   {i+1:2d}. {file.name:<40s} ({size_mb:6.1f}MB, {mod_time})")
        
        if len(checkpoint_files) > 15:
            print(f"   ... and {len(checkpoint_files) - 15} more")
        
        # Interactive selection
        try:
            print(f"\n{'='*70}")
            selection = input(f"Enter file number to analyze (1-{len(checkpoint_files)}), or 'all' to compare: ")
            
            if selection.lower() == 'all':
                args.checkpoint = [str(f) for f in checkpoint_files]
                args.compare = True
            else:
                idx = int(selection) - 1
                if 0 <= idx < len(checkpoint_files):
                    args.checkpoint = [str(checkpoint_files[idx])]
                else:
                    print("Invalid selection")
                    return
            
            # Ask for analysis type
            if len(args.checkpoint) == 1:
                print(f"\n{'='*70}")
                print("📊 Analysis Options:")
                print(f"{'='*70}")
                print("   1. Text analysis only (quick)")
                print("   2. Comprehensive dashboard (recommended)")
                print("   3. Full report (all visualizations + text)")
                print("   4. Custom (choose specific analyses)")
                print(f"{'='*70}")
                
                choice = input("\nSelect option (1-4, default=3): ").strip() or "3"
                
                if choice == "2":
                    args.dashboard = True
                elif choice == "3":
                    args.full_report = True
                elif choice == "4":
                    print("\n📊 Select analyses to generate:")
                    if input("   Dashboard? (y/n): ").lower() == 'y':
                        args.dashboard = True
                    if input("   Architecture? (y/n): ").lower() == 'y':
                        args.architecture = True
                    if input("   Stability? (y/n): ").lower() == 'y':
                        args.stability = True
                    if input("   Per-class? (y/n): ").lower() == 'y':
                        args.per_class = True
                
                # Ask about output directory
                print(f"\n📁 Default output directory: {default_output}")
                use_default = input("Use default? (y/n, default=y): ").strip().lower() != 'n'
                
                if not use_default:
                    custom_dir = input("Enter output directory: ").strip()
                    if custom_dir:
                        output_dir = Path(custom_dir)
                    
        except (ValueError, KeyboardInterrupt):
            print("\nAborted")
            return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"{'='*70}")
    
    # Create analyzer
    analyzer = EnhancedCheckpointAnalyzer()
    
    # Compare mode
    if args.compare and len(args.checkpoint) > 1:
        print(f"\n{'='*70}")
        print("COMPARING MULTIPLE CHECKPOINTS")
        print(f"{'='*70}")
        
        comparison_df = analyzer.compare_checkpoints(args.checkpoint)
        if not comparison_df.empty:
            comparison_file = output_dir / 'checkpoint_comparison.csv'
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\n✓ Comparison saved to {comparison_file}")
            
            # Also save as formatted text
            comparison_text_file = output_dir / 'checkpoint_comparison.txt'
            with open(comparison_text_file, 'w') as f:
                f.write("="*70 + "\n")
                f.write("CHECKPOINT COMPARISON\n")
                f.write("="*70 + "\n\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n" + "="*70 + "\n")
            print(f"✓ Comparison text saved to {comparison_text_file}")
        return
    
    # Single checkpoint analysis
    checkpoint_path = args.checkpoint[0]
    
    try:
        print(f"\n{'='*70}")
        print(f"ANALYZING CHECKPOINT")
        print(f"{'='*70}")
        print(f"File: {checkpoint_path}")
        
        checkpoint = analyzer.load_checkpoint(checkpoint_path)
        
        if not checkpoint:
            print("Failed to load checkpoint")
            return
        
        # Run analysis
        analysis = analyzer.analyze_checkpoint(checkpoint)
        
        # Generate outputs based on flags
        if args.full_report:
            print(f"\n{'='*70}")
            print("GENERATING FULL REPORT")
            print(f"{'='*70}")
            analyzer.generate_comprehensive_report(checkpoint, output_dir)
        else:
            # Generate individual analyses
            base_name = Path(checkpoint_path).stem
            
            if args.dashboard:
                print(f"\n🎨 Generating comprehensive dashboard...")
                analyzer.plot_comprehensive_dashboard(
                    checkpoint,
                    save_path=output_dir / f'{base_name}_dashboard.png'
                )
            
            if args.architecture:
                print(f"\n🏗️ Generating architecture visualization...")
                analyzer.plot_model_architecture(
                    checkpoint,
                    save_path=output_dir / f'{base_name}_architecture.png'
                )
            
            if args.stability:
                print(f"\n📈 Generating stability analysis...")
                analyzer.plot_training_stability(
                    checkpoint,
                    save_path=output_dir / f'{base_name}_stability.png'
                )
            
            if args.per_class:
                print(f"\n🎯 Generating per-class analysis...")
                analyzer.plot_per_class_performance(
                    checkpoint,
                    save_path=output_dir / f'{base_name}_per_class.png'
                )
            
            # If no specific flags, show text report and save it
            if not any([args.dashboard, args.architecture, args.stability, args.per_class]):
                report = analyzer._generate_text_report(checkpoint)
                print(f"\n{report}")
                
                # Save text report
                report_file = output_dir / f'{base_name}_report.txt'
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"\n✓ Report saved to {report_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        
        # List generated files
        generated_files = list(output_dir.glob(f'{Path(checkpoint_path).stem}*'))
        if generated_files:
            print(f"\n📄 Generated files:")
            for file in sorted(generated_files):
                size_kb = file.stat().st_size / 1e3
                print(f"   - {file.name} ({size_kb:.1f}KB)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()