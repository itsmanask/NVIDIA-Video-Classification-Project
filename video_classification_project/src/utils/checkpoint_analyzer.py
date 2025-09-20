import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class CheckpointAnalyzer:
    """Comprehensive analyzer for PyTorch checkpoint files"""
    
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
            # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
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
        
        print(f"\n{'='*60}")
        print(f"CHECKPOINT ANALYSIS")
        print(f"{'='*60}")
        
        # Basic Information
        print(f"\nBASIC INFORMATION")
        print(f"{'-'*40}")
        
        basic_info = {}
        
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            basic_info['epoch'] = epoch
            print(f"   Epoch: {epoch}")
        
        if 'timestamp' in checkpoint:
            timestamp = checkpoint['timestamp']
            basic_info['timestamp'] = timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                basic_info['formatted_time'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                basic_info['days_ago'] = (datetime.now() - dt.replace(tzinfo=None)).days
                print(f"   Created: {basic_info['formatted_time']}")
                print(f"   Age: {basic_info['days_ago']} days ago")
            except:
                print(f"   Created: {timestamp}")
        
        if 'error' in checkpoint:
            basic_info['error'] = checkpoint['error']
            basic_info['checkpoint_type'] = 'Emergency'
            print(f"   Type: Emergency Checkpoint")
            print(f"   Error: {checkpoint['error']}")
        else:
            basic_info['checkpoint_type'] = 'Normal'
            print(f"   Type: Normal Checkpoint")
        
        analysis['basic_info'] = basic_info
        
        # Model Information
        print(f"\nMODEL INFORMATION")
        print(f"{'-'*40}")
        
        model_info = {}
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Count parameters
            total_params = 0
            layer_info = []
            
            for name, param in state_dict.items():
                param_count = param.numel()
                total_params += param_count
                param_size_mb = param.numel() * 4 / 1e6  # Assuming float32
                
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
            
            for key, value in architecture_info.items():
                if key != 'layer_types':
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        analysis['model_info'] = model_info
        
        # Optimizer Information
        print(f"\nOPTIMIZER INFORMATION")
        print(f"{'-'*40}")
        
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
                            print(f"   {key.title()}: {value}")
            
            if 'state' in opt_state:
                optimizer_info['num_param_groups'] = len(opt_state['state'])
                print(f"   Parameter Groups: {len(opt_state['state'])}")
        
        analysis['optimizer_info'] = optimizer_info
        
        # Training Metrics
        print(f"\nTRAINING METRICS")
        print(f"{'-'*40}")
        
        metrics = {}
        
        if 'metrics' in checkpoint:
            checkpoint_metrics = checkpoint['metrics']
            
            # Current epoch metrics
            current_metrics = {}
            for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'best_val_acc']:
                if key in checkpoint_metrics:
                    value = checkpoint_metrics[key]
                    current_metrics[key] = value
                    
                    if 'acc' in key:
                        print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value:.4f}")
            
            metrics['current'] = current_metrics
            
            # Training history
            if 'history' in checkpoint_metrics:
                history = checkpoint_metrics['history']
                metrics['history'] = history
                
                print(f"\n   Training History:")
                for key, values in history.items():
                    if values:
                        latest = values[-1]
                        best_idx = np.argmax(values) if 'acc' in key else np.argmin(values)
                        best = values[best_idx]
                        
                        if 'acc' in key:
                            print(f"     {key}: Latest={latest:.2f}%, Best={best:.2f}% (epoch {best_idx})")
                        else:
                            print(f"     {key}: Latest={latest:.4f}, Best={best:.4f} (epoch {best_idx})")
        
        analysis['metrics'] = metrics
        
        # File Information
        if self.checkpoint_path:
            file_info = {
                'path': str(self.checkpoint_path),
                'size_mb': self.checkpoint_path.stat().st_size / 1e6,
                'modified': datetime.fromtimestamp(self.checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            analysis['file_info'] = file_info
            
            print(f"\nFILE INFORMATION")
            print(f"{'-'*40}")
            print(f"   Path: {file_info['path']}")
            print(f"   Size: {file_info['size_mb']:.2f}MB")
            print(f"   Modified: {file_info['modified']}")
        
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
            # Count layer types
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
    
    def compare_checkpoints(self, checkpoint_paths: List[str]) -> pd.DataFrame:
        """Compare multiple checkpoints"""
        print(f"\n{'='*60}")
        print(f"CHECKPOINT COMPARISON")
        print(f"{'='*60}")
        
        comparisons = []
        
        for path in checkpoint_paths:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                analysis = self.analyze_checkpoint(checkpoint)
                
                comparison = {
                    'file': Path(path).name,
                    'epoch': analysis['basic_info'].get('epoch', 'N/A'),
                    'type': analysis['basic_info'].get('checkpoint_type', 'Unknown'),
                    'age_days': analysis['basic_info'].get('days_ago', 'N/A'),
                    'total_params': analysis['model_info'].get('total_parameters', 0),
                    'model_size_mb': analysis['model_info'].get('total_size_mb', 0),
                    'train_acc': analysis['metrics']['current'].get('train_acc', 'N/A') if 'current' in analysis['metrics'] else 'N/A',
                    'val_acc': analysis['metrics']['current'].get('val_acc', 'N/A') if 'current' in analysis['metrics'] else 'N/A',
                    'best_val_acc': analysis['metrics']['current'].get('best_val_acc', 'N/A') if 'current' in analysis['metrics'] else 'N/A',
                    'train_loss': analysis['metrics']['current'].get('train_loss', 'N/A') if 'current' in analysis['metrics'] else 'N/A',
                    'val_loss': analysis['metrics']['current'].get('val_loss', 'N/A') if 'current' in analysis['metrics'] else 'N/A',
                }
                
                comparisons.append(comparison)
                print(f"✓ Analyzed {Path(path).name}")
                
            except Exception as e:
                print(f"✗ Failed to analyze {Path(path).name}: {e}")
                continue
        
        if comparisons:
            df = pd.DataFrame(comparisons)
            print(f"\nComparison Summary:")
            print(df.to_string(index=False))
            return df
        
        return pd.DataFrame()
    
    def get_available_metrics(self, checkpoint: Optional[Dict] = None) -> List[str]:
        """Get list of available metrics in the checkpoint history"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
            
        if not checkpoint or 'metrics' not in checkpoint:
            return []
            
        metrics = checkpoint['metrics']
        if 'history' not in metrics:
            return []
            
        history = metrics['history']
        return [k for k, v in history.items() if v and len(v) > 0]
    
    def plot_all_epochs_progression(self, checkpoint: Optional[Dict] = None, save_path: Optional[str] = None, 
                                   figsize: tuple = (25, 18)):
        """Plot all metrics showing progression from epoch 0 to current epoch"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint or 'metrics' not in checkpoint:
            print("No metrics data available for plotting")
            return
        
        metrics = checkpoint['metrics']
        
        if 'history' not in metrics or not metrics['history']:
            print("No training history available for plotting")
            return
        
        history = metrics['history']
        available_metrics = self.get_available_metrics(checkpoint)
        
        if not available_metrics:
            print("No training history data available")
            return
        
        current_epoch = checkpoint.get('epoch', len(next(iter(history.values()))) - 1)
        print(f"Plotting epoch progression from 0 to {current_epoch}...")
        print(f"Available metrics: {', '.join(available_metrics)}")
        
        # Set style for better visualization
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive plot showing all epochs
        fig = plt.figure(figsize=figsize)
        
        # Calculate subplot layout based on number of unique metric types
        loss_metrics = [m for m in available_metrics if 'loss' in m.lower()]
        acc_metrics = [m for m in available_metrics if 'acc' in m.lower()]
        
        # Create subplots
        if loss_metrics and acc_metrics:
            # Both loss and accuracy available
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Loss plot spans top row
            ax2 = fig.add_subplot(gs[1, :])  # Accuracy plot spans bottom row
        else:
            # Only one type available
            gs = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = None
        
        fig.suptitle(f'Training Progress: All Epochs (0 to {current_epoch})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Plot Loss metrics
        if loss_metrics:
            ax1.set_title('Loss Progression Across All Epochs', fontsize=14, fontweight='bold', pad=20)
            
            for i, metric in enumerate(loss_metrics):
                if metric in history and history[metric]:
                    epochs = list(range(len(history[metric])))
                    values = history[metric]
                    
                    # Different styles for train vs val
                    if 'train' in metric.lower():
                        linestyle = '-'
                        alpha = 0.8
                        linewidth = 2.5
                    elif 'val' in metric.lower():
                        linestyle = '--'
                        alpha = 0.9
                        linewidth = 3
                    else:
                        linestyle = '-.'
                        alpha = 0.7
                        linewidth = 2
                    
                    color = colors[i % len(colors)]
                    
                    ax1.plot(epochs, values, color=color, linestyle=linestyle, 
                            linewidth=linewidth, alpha=alpha, 
                            label=metric.replace('_', ' ').title(),
                            marker='o' if len(values) < 30 else None, markersize=4)
                    
                    # Add best value annotation
                    best_idx = np.argmin(values)
                    best_val = values[best_idx]
                    ax1.annotate(f'Best: {best_val:.4f}\n(Epoch {best_idx})', 
                               xy=(best_idx, best_val), xytext=(10, -20),
                               textcoords='offset points', fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
            
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')
            
            # Add current epoch marker
            ax1.axvline(x=current_epoch, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Current Epoch')
        
        # Plot Accuracy metrics
        if acc_metrics and ax2 is not None:
            ax2.set_title('Accuracy Progression Across All Epochs', fontsize=14, fontweight='bold', pad=20)
            
            for i, metric in enumerate(acc_metrics):
                if metric in history and history[metric]:
                    epochs = list(range(len(history[metric])))
                    values = history[metric]
                    
                    # Different styles for train vs val
                    if 'train' in metric.lower():
                        linestyle = '-'
                        alpha = 0.8
                        linewidth = 2.5
                    elif 'val' in metric.lower():
                        linestyle = '--'
                        alpha = 0.9
                        linewidth = 3
                    else:
                        linestyle = '-.'
                        alpha = 0.7
                        linewidth = 2
                    
                    color = colors[i % len(colors)]
                    
                    ax2.plot(epochs, values, color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha,
                            label=metric.replace('_', ' ').title(),
                            marker='o' if len(values) < 30 else None, markersize=4)
                    
                    # Add best value annotation
                    best_idx = np.argmax(values)
                    best_val = values[best_idx]
                    ax2.annotate(f'Best: {best_val:.2f}%\n(Epoch {best_idx})', 
                               xy=(best_idx, best_val), xytext=(10, 15),
                               textcoords='offset points', fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
            
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#f8f9fa')
            
            # Add current epoch marker
            ax2.axvline(x=current_epoch, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Current Epoch')
        
        elif acc_metrics and ax2 is None:
            # Only accuracy metrics available, use ax1
            ax1.set_title(f'Training Metrics Progression: Epochs 0-{current_epoch}', fontsize=14, fontweight='bold', pad=20)
            
            for i, metric in enumerate(acc_metrics):
                if metric in history and history[metric]:
                    epochs = list(range(len(history[metric])))
                    values = history[metric]
                    
                    # Different styles for train vs val
                    if 'train' in metric.lower():
                        linestyle = '-'
                        alpha = 0.8
                        linewidth = 2.5
                    elif 'val' in metric.lower():
                        linestyle = '--'
                        alpha = 0.9
                        linewidth = 3
                    else:
                        linestyle = '-.'
                        alpha = 0.7
                        linewidth = 2
                    
                    color = colors[i % len(colors)]
                    
                    ax1.plot(epochs, values, color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha,
                            label=metric.replace('_', ' ').title(),
                            marker='o' if len(values) < 30 else None, markersize=4)
                    
                    # Add best value annotation
                    if 'acc' in metric.lower():
                        best_idx = np.argmax(values)
                        best_val = values[best_idx]
                        ax1.annotate(f'Best: {best_val:.2f}%\n(Epoch {best_idx})', 
                                   xy=(best_idx, best_val), xytext=(10, 15),
                                   textcoords='offset points', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3),
                                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
            
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')
            
            # Add current epoch marker
            ax1.axvline(x=current_epoch, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Current Epoch')
        
        # Add summary text box
        summary_text = f"Training Summary:\n"
        summary_text += f"• Current Epoch: {current_epoch}\n"
        summary_text += f"• Total Metrics: {len(available_metrics)}\n"
        
        if loss_metrics and history.get('val_loss'):
            best_loss_epoch = np.argmin(history['val_loss'])
            best_loss = history['val_loss'][best_loss_epoch]
            summary_text += f"• Best Val Loss: {best_loss:.4f} (Epoch {best_loss_epoch})\n"
        
        if acc_metrics and history.get('val_acc'):
            best_acc_epoch = np.argmax(history['val_acc'])
            best_acc = history['val_acc'][best_acc_epoch]
            summary_text += f"• Best Val Acc: {best_acc:.2f}% (Epoch {best_acc_epoch})\n"
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
                verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"All epochs progression plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print summary statistics
    def plot_all_metrics(self, checkpoint: Optional[Dict] = None, save_path: Optional[str] = None, 
                        figsize: tuple = (25, 18)):
        """Plot all available metrics in a comprehensive dashboard"""
        # Call the new epoch progression plot instead
        self.plot_all_epochs_progression(checkpoint, save_path, figsize)
        """Plot all available metrics in a comprehensive dashboard"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint or 'metrics' not in checkpoint:
            print("No metrics data available for plotting")
            return
        
        metrics = checkpoint['metrics']
        
        if 'history' not in metrics or not metrics['history']:
            print("No training history available for plotting")
            return
        
        history = metrics['history']
        available_metrics = self.get_available_metrics(checkpoint)
        
        if not available_metrics:
            print("No training history data available")
            return
        
        print(f"Plotting comprehensive metrics dashboard...")
        print(f"Available metrics: {', '.join(available_metrics)}")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Categorize metrics
        accuracy_metrics = [m for m in available_metrics if 'acc' in m.lower() or 'accuracy' in m.lower()]
        loss_metrics = [m for m in available_metrics if 'loss' in m.lower()]
        f1_metrics = [m for m in available_metrics if 'f1' in m.lower()]
        precision_metrics = [m for m in available_metrics if 'precision' in m.lower()]
        recall_metrics = [m for m in available_metrics if 'recall' in m.lower()]
        lr_metrics = [m for m in available_metrics if 'lr' in m.lower() or 'learning_rate' in m.lower()]
        other_metrics = [m for m in available_metrics if m not in 
                        accuracy_metrics + loss_metrics + f1_metrics + 
                        precision_metrics + recall_metrics + lr_metrics]
        
        # Calculate subplot layout
        metric_groups = []
        if accuracy_metrics: metric_groups.append(('Accuracy', accuracy_metrics))
        if loss_metrics: metric_groups.append(('Loss', loss_metrics))
        if f1_metrics: metric_groups.append(('F1 Score', f1_metrics))
        if precision_metrics: metric_groups.append(('Precision', precision_metrics))
        if recall_metrics: metric_groups.append(('Recall', recall_metrics))
        if lr_metrics: metric_groups.append(('Learning Rate', lr_metrics))
        if other_metrics: metric_groups.append(('Other Metrics', other_metrics))
        
        n_groups = len(metric_groups)
        if n_groups == 0:
            print("No metrics to plot")
            return
        
        # Create subplots
        cols = min(3, n_groups)
        rows = (n_groups + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Training Metrics Dashboard - Epoch {checkpoint.get("epoch", "Unknown")}', 
                    fontsize=20, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        for idx, (group_name, group_metrics) in enumerate(metric_groups):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            for i, metric_name in enumerate(group_metrics):
                if metric_name not in history or not history[metric_name]:
                    continue
                    
                values = history[metric_name]
                epochs = range(len(values))
                
                # Choose color and style
                color = colors[i % len(colors)]
                linestyle = '-' if 'train' in metric_name else '--' if 'val' in metric_name else '-.'
                linewidth = 2.5 if 'val' in metric_name else 2
                alpha = 0.9 if 'val' in metric_name else 0.7
                
                ax.plot(epochs, values, color=color, linestyle=linestyle, 
                       linewidth=linewidth, alpha=alpha, label=metric_name.replace('_', ' ').title(),
                       marker='o' if len(values) < 50 else None, markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(group_name, fontsize=12)
            ax.set_title(f'{group_name} Over Training', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            # Add best value annotation for the last metric in group
            if group_metrics:
                last_metric = group_metrics[-1]
                if last_metric in history and history[last_metric]:
                    values = history[last_metric]
                    if 'loss' in last_metric.lower():
                        best_idx = np.argmin(values)
                        best_val = values[best_idx]
                        ax.annotate(f'Best: {best_val:.4f}', 
                                  xy=(best_idx, best_val), xytext=(10, 10),
                                  textcoords='offset points', 
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                    else:
                        best_idx = np.argmax(values)
                        best_val = values[best_idx]
                        ax.annotate(f'Best: {best_val:.4f}', 
                                  xy=(best_idx, best_val), xytext=(10, -15),
                                  textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Hide empty subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comprehensive metrics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Create additional detailed plots
        self._plot_metric_trends(checkpoint, save_path)
    
    def _plot_metric_trends(self, checkpoint: Dict, base_save_path: Optional[str] = None):
        """Plot detailed trend analysis for metrics"""
        history = checkpoint['metrics']['history']
        available_metrics = self.get_available_metrics(checkpoint)
        
        if len(available_metrics) < 2:
            return
        
        print("Creating detailed trend analysis plots...")
        
        # 1. Learning curve comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Training Analysis', fontsize=16, fontweight='bold')
        
        # Loss comparison
        loss_metrics = [m for m in available_metrics if 'loss' in m.lower()]
        if loss_metrics:
            for metric in loss_metrics:
                epochs = range(len(history[metric]))
                ax1.plot(epochs, history[metric], label=metric.replace('_', ' ').title(), 
                        linewidth=2, alpha=0.8)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Accuracy comparison
        acc_metrics = [m for m in available_metrics if 'acc' in m.lower()]
        if acc_metrics:
            for metric in acc_metrics:
                epochs = range(len(history[metric]))
                ax2.plot(epochs, history[metric], label=metric.replace('_', ' ').title(), 
                        linewidth=2, alpha=0.8)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy Curves')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Training vs Validation gap
        train_acc = None
        val_acc = None
        for metric in acc_metrics:
            if 'train' in metric:
                train_acc = history[metric]
            elif 'val' in metric:
                val_acc = history[metric]
        
        if train_acc and val_acc:
            min_len = min(len(train_acc), len(val_acc))
            gap = [train_acc[i] - val_acc[i] for i in range(min_len)]
            epochs = range(min_len)
            ax3.plot(epochs, gap, 'r-', linewidth=2, alpha=0.7)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Training - Validation Gap')
            ax3.set_title('Overfitting Analysis (Acc Gap)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Smoothed metrics (if enough data)
        if len(available_metrics) > 0:
            metric = available_metrics[0]
            if len(history[metric]) > 10:
                # Simple moving average
                window = min(5, len(history[metric]) // 4)
                smoothed = pd.Series(history[metric]).rolling(window=window, center=True).mean()
                epochs = range(len(history[metric]))
                ax4.plot(epochs, history[metric], alpha=0.3, label='Raw')
                ax4.plot(epochs, smoothed, linewidth=2, label=f'Smoothed (window={window})')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel(metric.replace('_', ' ').title())
                ax4.set_title('Smoothed Trend')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if base_save_path:
            if isinstance(base_save_path, str):
                trend_path = base_save_path.replace('.png', '_trends.png')
            else:
                trend_path = str(base_save_path).replace('.png', '_trends.png')
            plt.savefig(trend_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Trend analysis plot saved to {trend_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_curves(self, checkpoint: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot training curves from checkpoint history (backward compatibility)"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint or 'metrics' not in checkpoint:
            print("No metrics data available for plotting")
            return
        
        metrics = checkpoint['metrics']
        
        if 'history' not in metrics or not metrics['history']:
            print("No training history available for plotting")
            return
        
        history = metrics['history']
        
        # Check if we have data to plot
        available_metrics = [k for k, v in history.items() if v]
        if not available_metrics:
            print("No training history data available")
            return
        
        print(f"Plotting training curves...")
        
        # Setup the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {checkpoint.get("epoch", "Unknown")}', fontsize=16)
        
        # Loss plots
        if 'train_loss' in history and history['train_loss']:
            epochs = range(len(history['train_loss']))
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
            if 'val_loss' in history and history['val_loss']:
                axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
            
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plots
        if 'train_acc' in history and history['train_acc']:
            epochs = range(len(history['train_acc']))
            axes[0, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
            if 'val_acc' in history and history['val_acc']:
                axes[0, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
            
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss trend (recent epochs)
        if 'val_loss' in history and len(history['val_loss']) > 10:
            recent_epochs = list(range(len(history['val_loss']) - 10, len(history['val_loss'])))
            recent_loss = history['val_loss'][-10:]
            
            axes[1, 0].plot(recent_epochs, recent_loss, 'g-', linewidth=2, marker='o')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Validation Loss')
            axes[1, 0].set_title('Recent Validation Loss Trend')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy improvement
        if 'val_acc' in history and history['val_acc']:
            val_acc = np.array(history['val_acc'])
            best_so_far = np.maximum.accumulate(val_acc)
            epochs = range(len(val_acc))
            
            axes[1, 1].plot(epochs, val_acc, 'b-', alpha=0.6, label='Val Acc')
            axes[1, 1].plot(epochs, best_so_far, 'r-', linewidth=2, label='Best So Far')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Validation Accuracy Progress')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def extract_best_metrics(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract best metrics from training history"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        if not checkpoint or 'metrics' not in checkpoint:
            return {}
        
        metrics = checkpoint['metrics']
        best_metrics = {}
        
        if 'history' in metrics and metrics['history']:
            history = metrics['history']
            
            for metric_name, values in history.items():
                if not values:
                    continue
                
                if 'acc' in metric_name or 'f1' in metric_name or 'precision' in metric_name or 'recall' in metric_name:
                    # Higher is better for accuracy, f1, precision, recall
                    best_idx = np.argmax(values)
                    best_value = values[best_idx]
                    best_metrics[f'best_{metric_name}'] = {
                        'value': best_value,
                        'epoch': best_idx,
                        'current': values[-1] if values else None
                    }
                elif 'loss' in metric_name:
                    # Lower is better for loss
                    best_idx = np.argmin(values)
                    best_value = values[best_idx]
                    best_metrics[f'best_{metric_name}'] = {
                        'value': best_value,
                        'epoch': best_idx,
                        'current': values[-1] if values else None
                    }
        
        return best_metrics
    
    def generate_report(self, checkpoint: Optional[Dict] = None, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive text report"""
        if checkpoint is None:
            checkpoint = self.checkpoint_data
        
        analysis = self.analyze_checkpoint(checkpoint)
        best_metrics = self.extract_best_metrics(checkpoint)
        
        report = []
        report.append("=" * 80)
        report.append("CHECKPOINT ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Basic info
        basic_info = analysis['basic_info']
        report.append(f"\nBASIC INFORMATION:")
        report.append(f"   File: {self.checkpoint_path.name if self.checkpoint_path else 'Unknown'}")
        report.append(f"   Type: {basic_info.get('checkpoint_type', 'Unknown')}")
        report.append(f"   Epoch: {basic_info.get('epoch', 'N/A')}")
        report.append(f"   Created: {basic_info.get('formatted_time', 'Unknown')}")
        
        if 'error' in basic_info:
            report.append(f"   ERROR: {basic_info['error']}")
        
        # Model info
        model_info = analysis['model_info']
        report.append(f"\nMODEL INFORMATION:")
        report.append(f"   Parameters: {model_info.get('total_parameters', 0):,}")
        report.append(f"   Model Size: {model_info.get('total_size_mb', 0):.2f}MB")
        report.append(f"   Layers: {model_info.get('layer_count', 0)}")
        
        # Current metrics
        if 'current' in analysis['metrics']:
            current = analysis['metrics']['current']
            report.append(f"\nCURRENT METRICS:")
            for key, value in current.items():
                if isinstance(value, float):
                    if 'acc' in key:
                        report.append(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
                    else:
                        report.append(f"   {key.replace('_', ' ').title()}: {value:.4f}")
        
        # Best metrics
        if best_metrics:
            report.append(f"\nBEST METRICS:")
            for metric_name, info in best_metrics.items():
                value = info['value']
                epoch = info['epoch']
                current = info.get('current')
                
                if 'acc' in metric_name or 'f1' in metric_name or 'precision' in metric_name or 'recall' in metric_name:
                    report.append(f"   {metric_name}: {value:.4f} (epoch {epoch})")
                    if current is not None:
                        report.append(f"      Current: {current:.4f}")
                else:
                    report.append(f"   {metric_name}: {value:.4f} (epoch {epoch})")
                    if current is not None:
                        report.append(f"      Current: {current:.4f}")
        
        # Available metrics summary
        available_metrics = self.get_available_metrics(checkpoint)
        if available_metrics:
            report.append(f"\nAVAILABLE METRICS:")
            report.append(f"   {', '.join(available_metrics)}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text


def main():
    """Main function to analyze checkpoints"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PyTorch checkpoint files')
    parser.add_argument('checkpoint', nargs='*', help='Path(s) to checkpoint file(s)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple checkpoints')
    parser.add_argument('--plot', action='store_true', help='Plot basic training curves')
    parser.add_argument('--plot-all', action='store_true', help='Plot comprehensive metrics dashboard')
    parser.add_argument('--metrics', action='store_true', help='Show available metrics without plotting')
    parser.add_argument('--report', type=str, help='Save detailed report to file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots and reports')
    parser.add_argument('--figsize', type=str, default='20,15', help='Figure size for plots (width,height)')
    
    args = parser.parse_args()
    
    if not args.checkpoint:
        # Interactive mode - find checkpoint files
        print("No checkpoint files specified. Searching for checkpoints...")
        
        # Common checkpoint locations
        search_paths = [
            Path('.'),
            Path('checkpoints'),
            Path('models'),
            Path('video_classification_project/models'),
            Path('video_classification_project/models/checkpoints')
        ]
        
        checkpoint_files = []
        seen_files = set()  # Track files to avoid duplicates
        
        for search_path in search_paths:
            if search_path.exists():
                found_files = list(search_path.glob('*.pt')) + list(search_path.glob('*checkpoint*.pt'))
                for file in found_files:
                    # Use resolved path to avoid duplicates from symlinks/relative paths
                    resolved_path = file.resolve()
                    if resolved_path not in seen_files:
                        checkpoint_files.append(file)
                        seen_files.add(resolved_path)
        
        if not checkpoint_files:
            print("No checkpoint files found!")
            print("Please specify checkpoint file(s) as arguments:")
            print("   python checkpoint_analyzer.py checkpoint1.pt [checkpoint2.pt ...]")
            return
        
        print(f"Found {len(checkpoint_files)} checkpoint file(s):")
        for i, file in enumerate(checkpoint_files):
            print(f"   {i+1}. {file}")
        
        # Let user select files
        try:
            selection = input(f"\nEnter file number(s) to analyze (1-{len(checkpoint_files)}, or 'all'): ")
            if selection.lower() == 'all':
                selected_files = checkpoint_files
            else:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_files = [checkpoint_files[i] for i in indices if 0 <= i < len(checkpoint_files)]
        except (ValueError, IndexError):
            print("Invalid selection. Analyzing first file.")
            selected_files = [checkpoint_files[0]]
        
        args.checkpoint = [str(f) for f in selected_files]
        
        # Interactive options menu
        if len(selected_files) == 1:
            print(f"\nSelected: {selected_files[0].name}")
            print("\nAnalysis Options:")
            print("   1. Basic analysis only (text report)")
            print("   2. Basic analysis + simple training curves")
            print("   3. All epochs progression plot (RECOMMENDED)")
            
            try:
                choice = input("\nSelect option (1-3, default=1): ").strip()
                if not choice:
                    choice = "1"
                
                if choice == "2":
                    args.plot = True
                elif choice == "3":
                    args.plot_all = True
                
                # Always use extra large figure size
                args.figsize = "25,18"
                        
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default analysis (text report only)")
        
        elif len(selected_files) > 1:
            print(f"\nSelected {len(selected_files)} files for comparison")
            args.compare = True
    
    analyzer = CheckpointAnalyzer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse figsize - default to extra large
    figsize = tuple(map(int, args.figsize.split(','))) if hasattr(args, 'figsize') and args.figsize else (25, 18)
    
    if args.compare and len(args.checkpoint) > 1:
        # Compare multiple checkpoints
        print("Comparing checkpoints...")
        comparison_df = analyzer.compare_checkpoints(args.checkpoint)
        
        if not comparison_df.empty:
            # Save comparison
            comparison_file = output_dir / 'checkpoint_comparison.csv'
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\nComparison saved to {comparison_file}")
    
    else:
        # Analyze single checkpoint
        checkpoint_path = args.checkpoint[0]
        print(f"Analyzing checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = analyzer.load_checkpoint(checkpoint_path)
            
            if not checkpoint:
                print("Failed to load checkpoint data")
                return
                
            analysis = analyzer.analyze_checkpoint(checkpoint)
            
            # Show available metrics if requested
            if args.metrics:
                available_metrics = analyzer.get_available_metrics(checkpoint)
                print(f"\nAvailable metrics in checkpoint:")
                for metric in available_metrics:
                    print(f"   - {metric}")
                print(f"\nTotal: {len(available_metrics)} metrics")
                return
            
            # Generate report
            if args.report:
                report_path = output_dir / args.report
                analyzer.generate_report(checkpoint, report_path)
            else:
                report = analyzer.generate_report(checkpoint)
                print(f"\n{report}")
            
            # Plot comprehensive metrics dashboard
            if args.plot_all:
                plot_path = output_dir / f"{Path(checkpoint_path).stem}_all_metrics.png"
                analyzer.plot_all_metrics(checkpoint, plot_path, figsize=figsize)
            
            # Plot basic training curves (backward compatibility)
            elif args.plot:
                plot_path = output_dir / f"{Path(checkpoint_path).stem}_training_curves.png"
                analyzer.plot_training_curves(checkpoint, plot_path)
            
        except Exception as e:
            print(f"Error analyzing checkpoint: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()