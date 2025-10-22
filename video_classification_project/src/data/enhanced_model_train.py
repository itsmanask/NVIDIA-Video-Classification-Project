class MemoryMonitor:
    """Monitor system memory usage"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_info(self):
        """Get detailed memory information"""
        mem = psutil.virtual_memory()
        gpu_mem = None
        
        if torch.cuda.is_available():
            gpu_mem = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        
        return {
            'ram': {
                'total': mem.total / 1e9,
                'available': mem.available / 1e9,
                'used': mem.used / 1e9,
                'percent': mem.percent
            },
            'gpu': gpu_mem,
            'process': self.process.memory_info().rss / 1e9
        }
    
    def get_available_memory_gb(self):
        """Get available RAM in GB"""
        return psutil.virtual_memory().available / 1e9
    
    def print_memory_status(self):
        """Print current memory status"""
        info = self.get_memory_info()
        
        print(f"\n💾 Memory Status:")
        print(f"   RAM: {info['ram']['used']:.1f}/{info['ram']['total']:.1f}GB ({info['ram']['percent']:.1f}%)")
        print(f"   Available: {info['ram']['available']:.1f}GB")
        print(f"   Process: {info['process']:.1f}GB")
        
        if info['gpu']:
            print(f"   GPU: {info['gpu']['allocated']:.1f}/{info['gpu']['total']:.1f}GB")


def main():
    """Main function for A100 server training"""
    print("="*80)
    print("ENHANCED VIDEO CLASSIFICATION TRAINER - A100 SERVER")
    print("Features: Temporal Augmentation | Class Balancing | Focal Loss")
    print("          Multi-head Attention | Ensemble Models | OneCycleLR")
    print("="*80)
    
    # Configuration for A100 server
    data_dir = Path("video_classification_project/data/processed")
    output_dir = Path("video_classification_project/models_enhanced")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure your data is preprocessed and available.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # A100 optimal configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize trainer
    trainer = EnhancedVideoTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        max_memory_gb=30.0,  # Conservative for A100 40GB
        gradient_accumulation_steps=1  # No need for accumulation with A100
    )
    
    try:
        # Train with all enhancements
        print(f"\n🚀 Starting enhanced training pipeline...")
        print(f"   Data directory: {data_dir}")
        print(f"   Output directory: {output_dir}")
        print(f"   Device: {device}")
        
        # You can toggle ensemble training here
        USE_ENSEMBLE = False  # Set to True for ensemble (requires more memory)
        
        model, results = trainer.train_with_all_enhancements(
            num_epochs=50,
            batch_size=16,  # A100 can handle larger batches
            warmup_epochs=5,  # Warmup for backbone fine-tuning
            use_ensemble=USE_ENSEMBLE
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Best validation accuracy: {results['best_val_metrics']['val_acc']:.2f}%")
        print(f"Best worst-class F1: {results['best_val_metrics']['worst_class_f1']:.2f}%")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU Out of Memory Error!")
        print(f"   Error: {str(e)}")
        print(f"\nSolutions:")
        print(f"   1. Reduce batch size (current: 16)")
        print(f"   2. Disable ensemble training")
        print(f"   3. Use gradient accumulation")
        print(f"   4. Reduce model size (hidden_dim, num_layers)")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print(f"Partial results saved to {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set environment variables for A100 optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Enable TF32 on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set high priority for better performance
    torch.set_num_threads(os.cpu_count())
    
    # Run main training
    main()
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
import os
import sys
import platform
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
import warnings
import pickle
import psutil
import gc
import h5py
import mmap
import random
from collections import Counter
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        print(f"   Initialized Label Smoothing with alpha={smoothing}")
    
    def forward(self, predictions, targets):
        log_probs = nn.functional.log_softmax(predictions, dim=-1)
        
        # True label loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # Smooth loss (entropy)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Label Smoothing Loss"""
    
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1, focal_weight=0.7):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing = LabelSmoothingCrossEntropy(num_classes, smoothing)
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smooth = self.label_smoothing(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth


class TemporalAugmentation:
    """Temporal augmentation strategies for video data"""
    
    def __init__(self, frame_sampling_rate=0.8, speed_rate=(0.5, 2.0), 
                 temporal_shift=True, frame_drop_rate=0.1):
        self.frame_sampling_rate = frame_sampling_rate
        self.speed_rate = speed_rate
        self.temporal_shift = temporal_shift
        self.frame_drop_rate = frame_drop_rate
    
    def __call__(self, video_tensor):
        """Apply temporal augmentations to video tensor
        video_tensor: [num_frames, channels, height, width]
        """
        num_frames = video_tensor.shape[0]
        
        # Random frame sampling
        if random.random() < self.frame_sampling_rate:
            sample_size = max(int(num_frames * random.uniform(0.7, 1.0)), 1)
            indices = sorted(random.sample(range(num_frames), sample_size))
            video_tensor = video_tensor[indices]
        
        # Speed variation (frame interpolation/decimation)
        if self.speed_rate is not None and random.random() < 0.5:
            speed_factor = random.uniform(*self.speed_rate)
            new_length = int(num_frames * speed_factor)
            new_length = max(new_length, 1)
            indices = np.linspace(0, num_frames - 1, new_length).astype(int)
            video_tensor = video_tensor[indices]
        
        # Frame dropping
        if self.frame_drop_rate > 0 and random.random() < 0.3:
            keep_prob = 1 - self.frame_drop_rate
            mask = torch.rand(len(video_tensor)) < keep_prob
            if mask.sum() > 0:
                video_tensor = video_tensor[mask]
        
        # Temporal shift
        if self.temporal_shift and random.random() < 0.3:
            shift = random.randint(-2, 2)
            if shift != 0:
                video_tensor = torch.roll(video_tensor, shifts=shift, dims=0)
        
        return video_tensor


class EnhancedVideoDataset(Dataset):
    """Enhanced dataset with temporal augmentation and class balancing"""
    
    def __init__(self, data_dir, split='train', max_memory_gb=10.0, cache_size=100,
                 temporal_aug=True, compute_class_weights=True):
        print(f"\n{'='*60}")
        print(f"INITIALIZING {split.upper()} DATASET (Enhanced Mode)")
        print(f"{'='*60}")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_memory_gb = max_memory_gb
        self.cache_size = cache_size
        self.is_training = (split == 'train')
        
        # Temporal augmentation for training only
        if temporal_aug and self.is_training:
            self.temporal_aug = TemporalAugmentation()
            print("   ✓ Temporal augmentation enabled")
        else:
            self.temporal_aug = None
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # File index instead of loading all data
        self.file_index = []
        self.labels_index = []
        self.total_samples = 0
        self.category_mapping = {}
        self.class_counts = None
        self.class_weights = None
        
        # LRU cache for recently accessed data
        self.cache = {}
        self.cache_order = []
        
        # Checkpoint file for resuming
        self.checkpoint_file = self.data_dir / f"{split}_dataset_index.pkl"
        
        # Try to load existing index
        if self.checkpoint_file.exists():
            print(f"Found existing index: {self.checkpoint_file}")
            try:
                self._load_checkpoint()
                print(f"Loaded index with {self.total_samples:,} samples")
                
                # Compute class statistics
                if compute_class_weights and self.is_training:
                    self._compute_class_statistics()
                
                return
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Rebuilding index...")
        
        # Build index
        self._build_index()
        
        # Compute class statistics for training
        if compute_class_weights and self.is_training:
            self._compute_class_statistics()
        
        # Save checkpoint
        self._save_checkpoint()
    
    def _compute_class_statistics(self):
        """Compute class counts and weights for balanced training"""
        if not self.labels_index:
            return
        
        # Count class frequencies
        self.class_counts = Counter(self.labels_index)
        num_classes = len(self.class_counts)
        total_samples = sum(self.class_counts.values())
        
        print(f"\nClass Distribution ({self.split}):")
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            percentage = (count / total_samples) * 100
            class_name = self._get_class_name(class_id)
            print(f"   Class {class_id} ({class_name}): {count:,} samples ({percentage:.1f}%)")
        
        # Compute balanced weights (inverse frequency)
        if self.is_training:
            weights = []
            for class_id in range(num_classes):
                if class_id in self.class_counts:
                    # Inverse frequency weighting with smoothing
                    weight = total_samples / (num_classes * self.class_counts[class_id])
                    # Clip extreme weights
                    weight = min(max(weight, 0.5), 10.0)
                else:
                    weight = 1.0
                weights.append(weight)
            
            self.class_weights = torch.FloatTensor(weights)
            print(f"\nClass Weights for Balanced Training:")
            for i, w in enumerate(weights):
                print(f"   Class {i} ({self._get_class_name(i)}): {w:.3f}")
    
    def _get_class_name(self, class_id):
        """Get class name from mapping"""
        if self.category_mapping:
            for name, id in self.category_mapping.items():
                if id == class_id:
                    return name
        return f"Unknown_{class_id}"
    
    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler"""
        if not self.class_weights or not self.labels_index:
            return None
        
        sample_weights = []
        for label in self.labels_index:
            sample_weights.append(self.class_weights[label].item())
        
        return sample_weights
    
    def _build_index(self):
        """Build an index of all data files without loading them"""
        print(f"\nBuilding dataset index...")
        print(f"Memory limit: {self.max_memory_gb:.1f}GB")
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            return
        
        # Scan for data files
        data_files = []
        category_dirs = sorted([d for d in split_dir.glob("*") if d.is_dir()])
        
        print(f"Found {len(category_dirs)} categories")
        
        # Collect all data files with size info
        with tqdm(total=len(category_dirs), desc="Scanning categories") as pbar:
            for category_dir in category_dirs:
                pbar.set_description(f"Scanning {category_dir.name}")
                subcat_dirs = sorted([d for d in category_dir.glob("*") if d.is_dir()])
                
                for subcat_dir in subcat_dirs:
                    # Look for processed_data.pt
                    data_file = subcat_dir / 'processed_data.pt'
                    if data_file.exists():
                        file_size_gb = data_file.stat().st_size / 1e9
                        data_files.append((data_file, file_size_gb))
                    else:
                        # Look for individual .pt files
                        pt_files = sorted(subcat_dir.glob("*.pt"))
                        for pt_file in pt_files:
                            file_size_gb = pt_file.stat().st_size / 1e9
                            data_files.append((pt_file, file_size_gb))
                
                pbar.update(1)
        
        if not data_files:
            print(f"No data files found!")
            return
        
        total_size_gb = sum(size for _, size in data_files)
        print(f"\nFound {len(data_files)} data files")
        print(f"Total size: {total_size_gb:.2f}GB")
        
        # Sort files by size for efficient loading
        data_files.sort(key=lambda x: x[1])
        
        # Build index
        print("\nBuilding file index...")
        start_time = time.time()
        
        with tqdm(total=len(data_files), desc="Indexing files") as pbar:
            for file_idx, (data_file, file_size_gb) in enumerate(data_files):
                try:
                    # Quick load to get metadata only
                    with torch.serialization._open_file_like(data_file, 'rb') as f:
                        storage = torch.load(f, map_location='cpu')
                        
                        if isinstance(storage, dict) and 'videos' in storage and 'labels' in storage:
                            if isinstance(storage['videos'], torch.Tensor):
                                num_samples = storage['videos'].shape[0]
                            else:
                                num_samples = len(storage['videos'])
                            
                            # Store file reference
                            self.file_index.append({
                                'path': data_file,
                                'start_idx': self.total_samples,
                                'end_idx': self.total_samples + num_samples,
                                'size_gb': file_size_gb,
                                'num_samples': num_samples
                            })
                            
                            # Store labels separately
                            if isinstance(storage['labels'], torch.Tensor):
                                self.labels_index.extend(storage['labels'].tolist())
                            else:
                                self.labels_index.extend(storage['labels'])
                            
                            # Update category mapping
                            if 'category_mapping' in storage:
                                self.category_mapping.update(storage['category_mapping'])
                            
                            self.total_samples += num_samples
                            
                            # Clear from memory
                            del storage
                            gc.collect()
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nFailed to index {data_file.name}: {e}")
                    pbar.update(1)
                    continue
        
        elapsed_time = time.time() - start_time
        print(f"\nIndexing completed in {timedelta(seconds=int(elapsed_time))}")
        print(f"Total samples indexed: {self.total_samples:,}")
    
    def _save_checkpoint(self):
        """Save index to disk for fast loading"""
        try:
            checkpoint_data = {
                'file_index': self.file_index,
                'labels_index': self.labels_index,
                'total_samples': self.total_samples,
                'category_mapping': self.category_mapping,
                'class_counts': dict(self.class_counts) if self.class_counts else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"Saved index checkpoint: {self.checkpoint_file}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load index from disk"""
        with open(self.checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.file_index = checkpoint_data['file_index']
        self.labels_index = checkpoint_data['labels_index']
        self.total_samples = checkpoint_data['total_samples']
        self.category_mapping = checkpoint_data['category_mapping']
        
        if 'class_counts' in checkpoint_data and checkpoint_data['class_counts']:
            self.class_counts = Counter(checkpoint_data['class_counts'])
        
        print(f"Index created: {checkpoint_data['timestamp']}")
    
    def _load_file_data(self, file_info):
        """Load data from file"""
        file_path = file_info['path']
        
        # Check cache first
        if str(file_path) in self.cache:
            return self.cache[str(file_path)]
        
        # Load data
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'videos' in data:
                videos = data['videos']
                
                # Add to cache with size limit
                if len(self.cache) >= self.cache_size:
                    # Remove least recently used
                    oldest = self.cache_order.pop(0)
                    del self.cache[oldest]
                    gc.collect()
                
                self.cache[str(file_path)] = videos
                self.cache_order.append(str(file_path))
                
                return videos
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        return None
    
    def _get_file_for_idx(self, idx):
        """Find which file contains the given index"""
        for file_info in self.file_index:
            if file_info['start_idx'] <= idx < file_info['end_idx']:
                return file_info
        return None
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range (0-{self.total_samples-1})")
        
        # Find the file containing this index
        file_info = self._get_file_for_idx(idx)
        if file_info is None:
            raise RuntimeError(f"Could not find file for index {idx}")
        
        # Load file data
        videos = self._load_file_data(file_info)
        
        if videos is None:
            raise RuntimeError(f"Failed to load data from {file_info['path']}")
        
        # Get the specific sample
        local_idx = idx - file_info['start_idx']
        video = videos[local_idx]
        label = self.labels_index[idx]
        
        # Apply temporal augmentation if in training mode
        if self.temporal_aug is not None:
            video = self.temporal_aug(video)
        
        return video, torch.tensor(label, dtype=torch.long)


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head temporal attention for video sequences"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        attended, weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attended))
        return x, weights


class TemporalPooling(nn.Module):
    """Advanced temporal pooling strategies"""
    
    def __init__(self, method='attention', hidden_dim=512):
        super().__init__()
        self.method = method
        
        if method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        
        if self.method == 'mean':
            return x.mean(dim=1)
        elif self.method == 'max':
            return x.max(dim=1)[0]
        elif self.method == 'attention':
            # Attention-weighted pooling
            weights = self.attention_weights(x)  # [batch, seq_len, 1]
            weights = F.softmax(weights, dim=1)
            pooled = (x * weights).sum(dim=1)
            return pooled
        else:
            # Last frame
            return x[:, -1, :]


class SuperEnhancedCNNLSTM(nn.Module):
    """Enhanced model with all proposed improvements"""
    
    def __init__(self, num_classes=4, hidden_dim=768, num_lstm_layers=3, 
                 num_attention_heads=8, dropout=0.3, backbone='resnet101',
                 bidirectional=True, use_attention=True, mlp_dims=[512, 256],
                 temporal_pooling='attention', freeze_backbone=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        
        # Enhanced CNN backbone selection
        print(f"\nInitializing Model Architecture:")
        print(f"   Backbone: {backbone}")
        
        if backbone == 'resnet50':
            self.cnn = models.resnet50(weights='IMAGENET1K_V2')
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif backbone == 'resnet101':
            self.cnn = models.resnet101(weights='IMAGENET1K_V2')
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif backbone == 'efficientnet_b4':
            self.cnn = models.efficientnet_b4(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
        elif backbone == 'efficientnet_v2_m':
            self.cnn = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Optionally freeze backbone initially
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
            print(f"   Backbone frozen (will unfreeze after warmup)")
        else:
            print(f"   Backbone trainable from start")
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stacked BiLSTM
        print(f"   LSTM: {num_lstm_layers} layers, hidden_dim={hidden_dim}, bidirectional={bidirectional}")
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Multi-head Temporal Attention
        if use_attention:
            print(f"   Attention: {num_attention_heads} heads")
            self.temporal_attention = MultiHeadTemporalAttention(
                lstm_output_dim, 
                num_heads=num_attention_heads,
                dropout=dropout
            )
        else:
            self.temporal_attention = None
        
        # Temporal Pooling
        print(f"   Temporal Pooling: {temporal_pooling}")
        self.temporal_pooling = TemporalPooling(temporal_pooling, lstm_output_dim)
        
        # Multi-layer MLP Classifier
        print(f"   MLP Classifier: {[lstm_output_dim] + mlp_dims + [num_classes]}")
        mlp_layers = []
        in_dim = lstm_output_dim
        
        for hidden_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*mlp_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("   Backbone unfrozen for fine-tuning")
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process frames through CNN
        x = x.view(-1, channels, height, width)
        
        # Use gradient checkpointing for memory efficiency
        if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
            features = torch.utils.checkpoint.checkpoint(self.cnn, x)
        else:
            features = self.cnn(x)
        
        features = features.view(batch_size * num_frames, -1)
        features = self.feature_projection(features)
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention if enabled
        if self.temporal_attention is not None:
            attended_features, _ = self.temporal_attention(lstm_out)
        else:
            attended_features = lstm_out
        
        # Temporal pooling
        pooled_features = self.temporal_pooling(attended_features)
        
        # Classification
        output = self.classifier(pooled_features)
        return output


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models with different architectures"""
    
    def __init__(self, models, weights=None, meta_learner=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
        # Optional meta-learner
        if meta_learner:
            num_models = len(models)
            num_classes = models[0].num_classes
            self.meta_learner = nn.Sequential(
                nn.Linear(num_models * num_classes, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:
            self.meta_learner = None
    
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            with torch.no_grad() if not model.training else torch.enable_grad():
                out = model(x)
                outputs.append(out)
        
        # Stack outputs: [batch, num_models, num_classes]
        outputs = torch.stack(outputs, dim=1)
        
        if self.meta_learner:
            # Concatenate all model outputs
            combined = outputs.view(outputs.shape[0], -1)
            return self.meta_learner(combined)
        else:
            # Weighted average
            if self.weights.device != outputs.device:
                self.weights = self.weights.to(outputs.device)
            
            # Apply softmax to each model's output
            probs = F.softmax(outputs, dim=-1)
            # Weighted average
            weighted_probs = probs * self.weights.view(1, -1, 1)
            return weighted_probs.sum(dim=1)


class EnhancedVideoTrainer:
    """Enhanced trainer for A100 server with all improvements"""
    
    def __init__(self, data_dir, output_dir, device='cuda', 
                 max_memory_gb=30.0, gradient_accumulation_steps=1):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_memory_gb = max_memory_gb
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Memory monitor
        self.memory_monitor = MemoryMonitor()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir / 'checkpoints')
        
        # Setup device optimizations
        self._setup_device_optimizations()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_per_class_f1': [],
            'learning_rates': []
        }
        
        # Best metrics tracking
        self.best_metrics = {
            'val_acc': 0,
            'val_f1': 0,
            'worst_class_f1': 0,
            'epoch': 0
        }
    
    def _setup_device_optimizations(self):
        """Setup GPU optimizations for A100"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n🖥️ GPU Configuration (A100 Server)")
            print(f"   Device: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   CUDA: {torch.version.cuda}")
            
            # Enable mixed precision for A100
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
            
            # A100 specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Use larger memory fraction for A100
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            print(f"   ✓ Mixed Precision (FP16/BF16) enabled")
            print(f"   ✓ TF32 enabled for A100 Tensor Cores")
            print(f"   ✓ Gradient accumulation: {self.gradient_accumulation_steps} steps")
        else:
            print(f"\nUsing CPU")
            self.scaler = None
            self.autocast = lambda: torch.enable_grad()
    
    def create_data_loaders(self, batch_size=16, num_workers=8):
        """Create data loaders optimized for A100 server"""
        print(f"\n{'='*60}")
        print(f"CREATING DATA LOADERS (A100 Optimized)")
        print(f"{'='*60}")
        
        # Show current memory status
        self.memory_monitor.print_memory_status()
        
        # Create datasets with higher memory limits for A100
        train_dataset = EnhancedVideoDataset(
            self.data_dir, 
            split='train',
            max_memory_gb=self.max_memory_gb,
            cache_size=200,  # Larger cache for 251GB RAM
            temporal_aug=True,
            compute_class_weights=True
        )
        
        val_dataset = EnhancedVideoDataset(
            self.data_dir,
            split='val',
            max_memory_gb=self.max_memory_gb,
            cache_size=100,
            temporal_aug=False,
            compute_class_weights=False
        )
        
        test_dataset = EnhancedVideoDataset(
            self.data_dir,
            split='test',
            max_memory_gb=self.max_memory_gb,
            cache_size=100,
            temporal_aug=False,
            compute_class_weights=False
        )
        
        # Get sample weights for balanced training
        sample_weights = train_dataset.get_sample_weights()
        
        # Create weighted sampler for training
        if sample_weights and len(sample_weights) > 0:
            weighted_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            print(f"   ✓ Using WeightedRandomSampler for balanced training")
            shuffle = False  # Don't shuffle when using sampler
        else:
            weighted_sampler = None
            shuffle = True
        
        # A100 can handle much larger batch sizes
        print(f"\nBatch Configuration:")
        print(f"   Batch size per GPU: {batch_size}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        print(f"   Number of workers: {num_workers}")
        
        # Create loaders with optimized settings for A100
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=weighted_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers // 2,
            pin_memory=True
        )
        
        print(f"\nDataset Summary:")
        print(f"   Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
        print(f"   Val: {len(val_dataset):,} samples ({len(val_loader)} batches)")
        print(f"   Test: {len(test_dataset):,} samples ({len(test_loader)} batches)")
        
        # Store class weights for loss function
        self.class_weights = train_dataset.class_weights
        
        return train_loader, val_loader, test_loader
    
    def compute_metrics(self, outputs, labels, num_classes=4):
        """Compute detailed metrics including per-class F1"""
        _, predicted = outputs.max(1)
        
        # Overall accuracy
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        accuracy = 100. * correct / total
        
        # F1 scores
        labels_np = labels.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        
        # Per-class and average F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, predicted_np, 
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )
        
        # Weighted average F1
        avg_f1 = f1_score(labels_np, predicted_np, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': avg_f1 * 100,
            'f1_per_class': f1 * 100,
            'precision_per_class': precision * 100,
            'recall_per_class': recall * 100
        }
    
    def train_epoch(self, model, loader, criterion, optimizer, epoch, scheduler=None):
        """Enhanced training with detailed metrics"""
        model.train()
        
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        # Reset gradient accumulation
        optimizer.zero_grad()
        
        # Progress tracking
        batch_times = []
        start_time = time.time()
        
        with tqdm(total=len(loader), desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (videos, labels) in enumerate(loader):
                batch_start = time.time()
                
                # Move to device
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with self.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # Step scheduler if using OneCycleLR
                    if scheduler and hasattr(scheduler, 'step') and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                
                # Statistics
                running_loss += loss.item() * self.gradient_accumulation_steps
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                # Update progress bar
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Calculate metrics for display
                if len(all_outputs) > 0:
                    batch_outputs = torch.cat(all_outputs)
                    batch_labels = torch.cat(all_labels)
                    metrics = self.compute_metrics(batch_outputs, batch_labels)
                    
                    pbar.set_postfix({
                        'loss': f'{running_loss/(batch_idx+1):.4f}',
                        'acc': f'{metrics["accuracy"]:.2f}%',
                        'f1': f'{metrics["f1_weighted"]:.2f}%',
                        'gpu': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                    })
                
                pbar.update(1)
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Final metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        final_metrics = self.compute_metrics(all_outputs, all_labels)
        
        epoch_loss = running_loss / len(loader)
        
        return epoch_loss, final_metrics
    
    def validate(self, model, loader, criterion):
        """Validation with detailed metrics"""
        model.eval()
        
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(total=len(loader), desc="Validation") as pbar:
                for videos, labels in loader:
                    videos = videos.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward with mixed precision
                    with self.autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    all_outputs.append(outputs.cpu())
                    all_labels.append(labels.cpu())
                    
                    pbar.update(1)
        
        # Compute final metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        final_metrics = self.compute_metrics(all_outputs, all_labels)
        
        val_loss = running_loss / len(loader)
        
        return val_loss, final_metrics
    
    def create_ensemble_models(self, num_classes=4):
        """Create diverse models for ensemble"""
        models = []
        
        # Model 1: BiLSTM + Attention (ResNet101)
        model1 = SuperEnhancedCNNLSTM(
            num_classes=num_classes,
            hidden_dim=768,
            num_lstm_layers=3,
            num_attention_heads=8,
            dropout=0.3,
            backbone='resnet101',
            bidirectional=True,
            use_attention=True,
            mlp_dims=[512, 256],
            temporal_pooling='attention'
        )
        models.append(model1)
        
        # Model 2: BiLSTM + Multi-head Attention (EfficientNet)
        model2 = SuperEnhancedCNNLSTM(
            num_classes=num_classes,
            hidden_dim=512,
            num_lstm_layers=2,
            num_attention_heads=12,
            dropout=0.25,
            backbone='efficientnet_b4',
            bidirectional=True,
            use_attention=True,
            mlp_dims=[384, 192],
            temporal_pooling='attention'
        )
        models.append(model2)
        
        # Model 3: Deep LSTM with different pooling
        model3 = SuperEnhancedCNNLSTM(
            num_classes=num_classes,
            hidden_dim=1024,
            num_lstm_layers=4,
            num_attention_heads=8,
            dropout=0.35,
            backbone='resnet50',
            bidirectional=True,
            use_attention=True,
            mlp_dims=[512, 256, 128],
            temporal_pooling='mean'
        )
        models.append(model3)
        
        print(f"\nCreated {len(models)} diverse models for ensemble")
        return models
    
    def train_with_all_enhancements(self, num_epochs=50, batch_size=16, 
                                   warmup_epochs=5, use_ensemble=False):
        """Main training loop with all enhancements"""
        print(f"\n{'='*60}")
        print(f"ENHANCED TRAINING FOR A100 SERVER")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            batch_size=batch_size, 
            num_workers=8
        )
        
        if len(train_loader) == 0:
            print("No training data available!")
            return
        
        # Determine number of classes
        num_classes = len(train_loader.dataset.category_mapping) if train_loader.dataset.category_mapping else 4
        
        # Create model or ensemble
        if use_ensemble:
            print(f"\n📦 Creating Model Ensemble...")
            models = self.create_ensemble_models(num_classes)
            model = ModelEnsemble(models, meta_learner=True)
        else:
            print(f"\n🏗️ Creating Single Enhanced Model...")
            model = SuperEnhancedCNNLSTM(
                num_classes=num_classes,
                hidden_dim=768,
                num_lstm_layers=3,
                num_attention_heads=8,
                dropout=0.3,
                backbone='resnet101',
                bidirectional=True,
                use_attention=True,
                mlp_dims=[512, 256],
                temporal_pooling='attention',
                freeze_backbone=(warmup_epochs > 0)
            )
        
        model = model.to(self.device)
        
        # Loss function with class weights
        print(f"\n⚖️ Loss Function Configuration:")
        if self.class_weights is not None:
            # Combined Focal + Label Smoothing with class weights
            alpha = self.class_weights.to(self.device)
            criterion = CombinedLoss(
                num_classes=num_classes,
                alpha=alpha,
                gamma=2.0,
                smoothing=0.1,
                focal_weight=0.7
            )
            print(f"   ✓ Using Combined Loss (Focal + Label Smoothing)")
            print(f"   ✓ Class weights applied for imbalance")
        else:
            criterion = LabelSmoothingCrossEntropy(num_classes, smoothing=0.1)
            print(f"   ✓ Using Label Smoothing CrossEntropy")
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-3,  # Higher LR for A100
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler - OneCycleLR for better convergence
        total_steps = len(train_loader) * num_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        print(f"\n🎯 Training Configuration:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        print(f"   Scheduler: OneCycleLR")
        print(f"   Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Max LR: 5e-3")
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch}/{num_epochs-1}")
            
            # Unfreeze backbone after warmup
            if epoch == warmup_epochs and hasattr(model, 'unfreeze_backbone'):
                print("   🔓 Unfreezing backbone for fine-tuning...")
                model.unfreeze_backbone()
                # Adjust learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            # Train
            train_loss, train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, scheduler
            )
            
            # Validate
            val_loss, val_metrics = self.validate(model, val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_weighted'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_weighted'])
            self.history['val_per_class_f1'].append(val_metrics['f1_per_class'])
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Track best model based on worst-class F1
            worst_class_f1 = val_metrics['f1_per_class'].min()
            
            is_best = False
            if worst_class_f1 > self.best_metrics['worst_class_f1']:
                self.best_metrics['worst_class_f1'] = worst_class_f1
                self.best_metrics['val_acc'] = val_metrics['accuracy']
                self.best_metrics['val_f1'] = val_metrics['f1_weighted']
                self.best_metrics['epoch'] = epoch
                is_best = True
                print(f"   🏆 New best model! Worst-class F1: {worst_class_f1:.2f}%")
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.2f}%, F1={train_metrics['f1_weighted']:.2f}%")
            print(f"   Val: Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.2f}%, F1={val_metrics['f1_weighted']:.2f}%")
            print(f"   Per-class F1: {val_metrics['f1_per_class'].tolist()}")
            print(f"   Worst-class F1: {worst_class_f1:.2f}%")
            print(f"   Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'best_metrics': self.best_metrics,
                'history': self.history
            }
            
            if is_best:
                torch.save(checkpoint_data, self.output_dir / 'best_model.pt')
            
            # Early stopping based on worst-class F1
            if epoch - self.best_metrics['epoch'] > 15:
                print(f"\n⏹️ Early stopping triggered!")
                break
        
        # Final evaluation on test set
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        
        # Load best model
        best_checkpoint = torch.load(self.output_dir / 'best_model.pt')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_loss, test_metrics = self.validate(model, test_loader, criterion)
        
        print(f"\n🎯 Final Test Results:")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"   Test F1 (weighted): {test_metrics['f1_weighted']:.2f}%")
        print(f"   Per-class F1: {test_metrics['f1_per_class'].tolist()}")
        print(f"   Worst-class F1: {test_metrics['f1_per_class'].min():.2f}%")
        
        # Save results
        results = {
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_metrics.items()},
            'best_val_metrics': self.best_metrics,
            'training_history': self.history,
            'config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'use_ensemble': use_ensemble,
                'warmup_epochs': warmup_epochs
            }
        }
        
        with open(self.output_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Results saved to {self.output_dir}")
        
        return model, results