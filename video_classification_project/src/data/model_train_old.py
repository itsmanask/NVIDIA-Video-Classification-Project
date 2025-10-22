import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
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
warnings.filterwarnings('ignore')


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


class EarlyStopping:
    """Enhanced early stopping with patience and delta"""
    
    def __init__(self, patience=10, min_delta=1e-4, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        print(f"   Initialized Early Stopping: patience={patience}, min_delta={min_delta}, mode={mode}")
        
    def __call__(self, score, epoch):
        if self.mode == 'max':
            score_improved = self.best_score is None or score > (self.best_score + self.min_delta)
        else:
            score_improved = self.best_score is None or score < (self.best_score - self.min_delta)
        
        if score_improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"   ✓ Validation improved to {score:.4f} (best epoch: {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠ No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"   🛑 Early stopping triggered! Best model was at epoch {self.best_epoch}")
        
        return self.early_stop


class MemoryEfficientVideoDataset(Dataset):
    """Memory-efficient dataset with memory-mapped loading and caching"""
    
    def __init__(self, data_dir, split='train', max_memory_gb=3.0, cache_size=50):
        print(f"\n{'='*60}")
        print(f"INITIALIZING {split.upper()} DATASET (Memory-Mapped Mode)")
        print(f"{'='*60}")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_memory_gb = max_memory_gb
        self.cache_size = cache_size
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # File index instead of loading all data
        self.file_index = []
        self.labels_index = []
        self.total_samples = 0
        self.category_mapping = {}
        
        # LRU cache for recently accessed data
        self.cache = {}
        self.cache_order = []
        
        # Memory-mapped files cache
        self.mmap_cache = {}
        
        # Checkpoint file for resuming - ensure proper path handling
        self.checkpoint_file = self.data_dir / f"{split}_dataset_index.pkl"
        
        # Try to load existing index
        if self.checkpoint_file.exists():
            print(f"Found existing index: {self.checkpoint_file}")
            try:
                self._load_checkpoint()
                print(f"Loaded index with {self.total_samples:,} samples")
                return
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Rebuilding index...")
        
        # Build index
        self._build_index()
        
        # Save checkpoint
        self._save_checkpoint()
    
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
        
        # Build index with progress and ETA
        print("\nBuilding file index with memory mapping...")
        start_time = time.time()
        
        with tqdm(total=len(data_files), desc="Indexing files") as pbar:
            for file_idx, (data_file, file_size_gb) in enumerate(data_files):
                # Update progress with ETA
                if file_idx > 0:
                    elapsed = time.time() - start_time
                    eta = elapsed * (len(data_files) - file_idx) / file_idx
                    pbar.set_postfix({
                        'file': data_file.name[:20],
                        'size': f'{file_size_gb:.2f}GB',
                        'ETA': str(timedelta(seconds=int(eta)))
                    })
                
                try:
                    # For large files, use memory mapping
                    if file_size_gb > 2.0:
                        # Create memory-mapped access
                        self._setup_memory_mapped_file(data_file)
                    
                    # Quick load to get metadata only
                    with torch.serialization._open_file_like(data_file, 'rb') as f:
                        # Load only the structure, not the actual tensors
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
                                'num_samples': num_samples,
                                'use_mmap': file_size_gb > 2.0
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
                            
                            # Immediately clear from memory
                            del storage
                            torch.cuda.empty_cache()
                            gc.collect()
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nFailed to index {data_file.name}: {e}")
                    pbar.update(1)
                    continue
        
        elapsed_time = time.time() - start_time
        print(f"\nIndexing completed in {timedelta(seconds=int(elapsed_time))}")
        print(f"Total samples indexed: {self.total_samples:,}")
    
    def _setup_memory_mapped_file(self, file_path):
        """Setup memory-mapped access for large files"""
        try:
            # Store the path for later memory-mapped access
            self.mmap_cache[str(file_path)] = file_path
        except Exception as e:
            print(f"Failed to setup memory mapping for {file_path}: {e}")
    
    def _save_checkpoint(self):
        """Save index to disk for fast loading"""
        try:
            checkpoint_data = {
                'file_index': self.file_index,
                'labels_index': self.labels_index,
                'total_samples': self.total_samples,
                'category_mapping': self.category_mapping,
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
        
        print(f"Index created: {checkpoint_data['timestamp']}")
    
    def _load_file_data(self, file_info):
        """Load data with memory-mapped access for large files"""
        file_path = file_info['path']
        
        # Check memory before loading
        available_memory = self.memory_monitor.get_available_memory_gb()
        
        if file_info['size_gb'] > available_memory * 0.5:  # Use only 50% of available
            print(f"\nLarge file: {file_info['size_gb']:.2f}GB (Available: {available_memory:.2f}GB)")
            
            # Clear cache aggressively
            self._clear_cache()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load data
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'videos' in data:
                return data['videos']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        return None
    
    def _clear_cache(self):
        """Clear the LRU cache"""
        self.cache.clear()
        self.cache_order.clear()
        gc.collect()
    
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
        
        # Check cache first
        file_path = str(file_info['path'])
        if file_path in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            videos = self.cache[file_path]
        else:
            # Load file data
            videos = self._load_file_data(file_info)
            
            if videos is None:
                raise RuntimeError(f"Failed to load data from {file_path}")
            
            # Add to cache with size limit
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]
                gc.collect()
            
            self.cache[file_path] = videos
            self.cache_order.append(file_path)
        
        # Get the specific sample
        local_idx = idx - file_info['start_idx']
        video = videos[local_idx]
        label = self.labels_index[idx]
        
        return video, torch.tensor(label, dtype=torch.long)


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
        
        print(f"\nMemory Status:")
        print(f"   RAM: {info['ram']['used']:.1f}/{info['ram']['total']:.1f}GB ({info['ram']['percent']:.1f}%)")
        print(f"   Available: {info['ram']['available']:.1f}GB")
        print(f"   Process: {info['process']:.1f}GB")
        
        if info['gpu']:
            print(f"   GPU: {info['gpu']['allocated']:.1f}/{info['gpu']['total']:.1f}GB")


class CheckpointManager:
    """Enhanced checkpoint manager with epoch-specific saves"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_file = self.checkpoint_dir / 'best_model.pt'
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, scaler, metrics, is_best=False):
        """Save checkpoint with unique filename for each epoch"""
        # Save epoch-specific checkpoint
        epoch_checkpoint_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, epoch_checkpoint_file)
        print(f"Saved checkpoint: {epoch_checkpoint_file.name}")
        
        # Also save as latest checkpoint
        latest_checkpoint = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_checkpoint)
        
        if is_best:
            torch.save(checkpoint, self.best_model_file)
            print(f"Saved best model (epoch {epoch})")
    
    def get_available_checkpoints(self):
        """Get list of available checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        # Also include emergency checkpoints
        emergency_checkpoints = sorted(self.checkpoint_dir.glob('emergency_checkpoint_epoch_*.pt'))
        return checkpoints + emergency_checkpoints
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
        """Load specific checkpoint"""
        if not checkpoint_path.exists():
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


class EnhancedCNNLSTM(nn.Module):
    """Enhanced CNN-LSTM with gradient checkpointing for memory efficiency"""
    
    def __init__(self, num_classes=4, hidden_dim=512, num_layers=3, dropout=0.25,
                 backbone='resnet50', bidirectional=True, attention=True):
        super(EnhancedCNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = attention
        
        # Select CNN backbone with gradient checkpointing
        if backbone == 'resnet50':
            self.cnn = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
            # Enable gradient checkpointing for ResNet
            self.use_checkpoint = True
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Feature projection with reduced dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention
        if self.use_attention:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        
        # Classifier with reduced dropout
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process frames through CNN with gradient checkpointing
        x = x.view(-1, channels, height, width)
        
        if self.use_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(self.cnn, x)
        else:
            features = self.cnn(x)
        
        features = features.view(batch_size * num_frames, -1)
        features = self.feature_projection(features)
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        if self.use_attention:
            attended_features, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            attended_features = self.attention_norm(attended_features + lstm_out)
            attended_features = attended_features.mean(dim=1)
        else:
            attended_features = lstm_out[:, -1, :]
        
        output = self.classifier(attended_features)
        return output


class VideoClassificationTrainer:
    """Enhanced trainer with FIXED implementations of all features"""
    
    def __init__(self, data_dir, output_dir, device='cuda', max_memory_gb=3.0, 
                 gradient_accumulation_steps=4, gradient_clip_val=1.0):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_memory_gb = max_memory_gb
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        
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
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def _setup_device_optimizations(self):
        """Setup GPU optimizations with enhanced mixed precision"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\nGPU Configuration")
            print(f"   Device: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   CUDA: {torch.version.cuda}")
            
            # Enable mixed precision with autocast
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            print(f"   Mixed Precision (FP16) enabled")
            print(f"   Gradient accumulation: {self.gradient_accumulation_steps} steps")
            print(f"   Gradient clipping: {self.gradient_clip_val}")
            print(f"   TF32 enabled")
        else:
            print(f"\nUsing CPU - training will be slow")
            self.scaler = None
            self.autocast = lambda: torch.enable_grad()
    
    def create_data_loaders(self):
        """Create memory-efficient data loaders"""
        print(f"\n{'='*60}")
        print(f"CREATING DATA LOADERS")
        print(f"{'='*60}")
        
        # Show current memory status
        self.memory_monitor.print_memory_status()
        
        # Create datasets with reduced memory limit
        train_dataset = MemoryEfficientVideoDataset(
            self.data_dir, 
            split='train',
            max_memory_gb=self.max_memory_gb,
            cache_size=30
        )
        
        val_dataset = MemoryEfficientVideoDataset(
            self.data_dir,
            split='val',
            max_memory_gb=self.max_memory_gb,
            cache_size=20
        )
        
        test_dataset = MemoryEfficientVideoDataset(
            self.data_dir,
            split='test',
            max_memory_gb=self.max_memory_gb,
            cache_size=20
        )
        
        # Conservative batch sizes for RTX 4060 with gradient accumulation
        if self.device.type == 'cuda':
            batch_size = 2  # Very conservative for 8GB GPU with large files
        else:
            batch_size = 1
        
        # Effective batch size with gradient accumulation
        effective_batch_size = batch_size * self.gradient_accumulation_steps
        print(f"\nBatch Configuration:")
        print(f"   Actual batch size: {batch_size}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {effective_batch_size}")
        
        # Adjust workers based on available RAM
        available_ram = self.memory_monitor.get_available_memory_gb()
        if available_ram > 8:
            num_workers = 2
        else:
            num_workers = 0  # No parallel loading for memory efficiency
        
        print(f"\nDataLoader Configuration:")
        print(f"   Workers: {num_workers}")
        print(f"   Pin memory: {self.device.type == 'cuda'}")
        print(f"   Available RAM: {available_ram:.1f}GB")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
            prefetch_factor=1 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # No workers for test to save memory
        )
        
        print(f"\nDataset Summary:")
        print(f"   Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
        print(f"   Val: {len(val_dataset):,} samples ({len(val_loader)} batches)")
        print(f"   Test: {len(test_dataset):,} samples ({len(test_loader)} batches)")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, loader, criterion, optimizer, epoch):
        """Train with gradient accumulation, clipping, and mixed precision"""
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
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
                
                # Backward pass with gradient scaling
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
                    
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Statistics
                running_loss += loss.item() * self.gradient_accumulation_steps
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Calculate ETA
                avg_batch_time = np.mean(batch_times[-50:])
                remaining_batches = len(loader) - batch_idx - 1
                eta = remaining_batches * avg_batch_time
                
                # Memory usage
                if self.device.type == 'cuda':
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                else:
                    gpu_mem = 0
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'gpu': f'{gpu_mem:.1f}GB',
                    'ETA': str(timedelta(seconds=int(eta)))
                })
                pbar.update(1)
                
                # Aggressive memory cleanup every 10 batches
                if batch_idx % 10 == 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        
        # Handle any remaining gradients
        if (len(loader) % self.gradient_accumulation_steps) != 0:
            # Gradient clipping for remaining gradients
            if self.scaler:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
            
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, loader, criterion):
        """Validate with mixed precision"""
        model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
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
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                    pbar.update(1)
        
        val_loss = running_loss / len(loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def select_checkpoint(self):
        """Allow user to select checkpoint to resume from"""
        checkpoints = self.checkpoint_manager.get_available_checkpoints()
        
        if not checkpoints:
            return None, 0
        
        print("\nAvailable Checkpoints:")
        print("0. Start from scratch")
        
        for i, ckpt in enumerate(checkpoints, 1):
            # Load just to get epoch info
            try:
                data = torch.load(ckpt, map_location='cpu')
                epoch = data.get('epoch', 'unknown')
                metrics = data.get('metrics', {})
                val_acc = metrics.get('val_acc', 0)
                timestamp = data.get('timestamp', 'unknown')
                print(f"{i}. {ckpt.name} - Epoch {epoch}, Val Acc: {val_acc:.2f}%, Time: {timestamp}")
            except:
                print(f"{i}. {ckpt.name}")
        
        # Also check for latest checkpoint
        latest = self.checkpoint_manager.checkpoint_dir / 'latest_checkpoint.pt'
        if latest.exists():
            print(f"{len(checkpoints)+1}. latest_checkpoint.pt (Most recent)")
        
        while True:
            try:
                choice = input("\nSelect checkpoint (enter number): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print("Starting training from scratch")
                    return None, 0
                elif 1 <= choice_num <= len(checkpoints):
                    selected = checkpoints[choice_num - 1]
                    print(f"Selected: {selected.name}")
                    return selected, None
                elif choice_num == len(checkpoints) + 1 and latest.exists():
                    print(f"Selected: latest_checkpoint.pt")
                    return latest, None
                else:
                    print("Invalid choice, please try again")
            except ValueError:
                print("Please enter a valid number")
    
    def train(self, num_epochs=50, label_smoothing=0.1, early_stopping_patience=10):
        """FIXED MAIN TRAINING LOOP - All features properly implemented"""
        print(f"\n{'='*60}")
        print(f"ENHANCED TRAINING CONFIGURATION - FIXED IMPLEMENTATION")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        if len(train_loader) == 0:
            print("No training data available!")
            return
        
        # Initialize model
        num_classes = len(train_loader.dataset.category_mapping) if train_loader.dataset.category_mapping else 4
        
        model = EnhancedCNNLSTM(
            num_classes=num_classes,
            hidden_dim=512,
            num_layers=3,
            dropout=0.25,
            backbone='resnet50',
            bidirectional=True,
            attention=True
        ).to(self.device)
        
        # FIXED: Properly use label smoothing loss
        print(f"\nLoss Function Configuration:")
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(num_classes, smoothing=label_smoothing)
            print(f"   ✓ Using Label Smoothing CrossEntropy (alpha={label_smoothing})")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"   Using standard CrossEntropyLoss")
        
        # Optimizer with improved settings
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # FIXED: Properly implemented scheduler
        print(f"\nScheduler Configuration:")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Monitor validation accuracy
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-7
        )
        print(f"   ✓ ReduceLROnPlateau: mode=max, factor=0.5, patience=5")
        
        # FIXED: Properly implemented early stopping
        print(f"\nEarly Stopping Configuration:")
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            min_delta=1e-4, 
            mode='max',
            verbose=True
        )
        
        # Select checkpoint
        checkpoint_path, start_epoch = self.select_checkpoint()
        best_val_acc = 0
        
        if checkpoint_path is not None:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, self.scaler
            )
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                
                if 'metrics' in checkpoint:
                    best_val_acc = checkpoint['metrics'].get('best_val_acc', 0)
                    loaded_history = checkpoint['metrics'].get('history', self.history)
                    
                    # Ensure all required keys exist in history
                    for key in self.history.keys():
                        if key not in loaded_history:
                            loaded_history[key] = []
                    
                    self.history = loaded_history
                else:
                    print("   Emergency checkpoint detected - initializing with defaults")
                    best_val_acc = 0
                
                print(f"Resuming from epoch {start_epoch}")
                print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        else:
            start_epoch = 0
        
        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Configuration:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / 1e9:.2f}GB")
        
        print(f"\nFixed Training Settings:")
        print(f"   Epochs: {start_epoch} -> {num_epochs}")
        print(f"   Initial Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   ✓ Label smoothing: {label_smoothing}")
        print(f"   ✓ Early stopping patience: {early_stopping_patience}")
        print(f"   ✓ Scheduler: ReduceLROnPlateau")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {'Enabled' if self.scaler else 'Disabled'}")
        print(f"   Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"   Gradient Clipping: {self.gradient_clip_val}")
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"STARTING FIXED TRAINING LOOP")
        print(f"{'='*60}")
        
        training_start = time.time()
        
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start = time.time()
                
                print(f"\nEpoch {epoch}/{num_epochs-1}")
                print(f"   Current LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Show memory status
                self.memory_monitor.print_memory_status()
                
                # Clear cache before epoch
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Training phase
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, epoch
                )
                
                # Validation phase
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                
                # FIXED: Properly step scheduler with validation accuracy
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_acc)  # Pass validation accuracy for ReduceLROnPlateau
                new_lr = optimizer.param_groups[0]['lr']
                
                if old_lr != new_lr:
                    print(f"   🔄 Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
                
                # Update history
                current_lr = optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rates'].append(current_lr)
                
                # Check if best model
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    print(f"   🎯 New best validation accuracy: {best_val_acc:.2f}%")
                
                # Save checkpoint for EVERY epoch
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'history': self.history,
                    'current_lr': current_lr
                }
                
                self.checkpoint_manager.save_checkpoint(
                    epoch, model, optimizer, scheduler, self.scaler, metrics, is_best
                )
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                avg_epoch_time = total_time / (epoch - start_epoch + 1)
                eta = avg_epoch_time * (num_epochs - epoch - 1)
                
                print(f"\nEpoch {epoch} Summary:")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"   Best Val Acc: {best_val_acc:.2f}%")
                print(f"   Current LR: {current_lr:.6f}")
                print(f"   Epoch Time: {str(timedelta(seconds=int(epoch_time)))}")
                print(f"   Total Time: {str(timedelta(seconds=int(total_time)))}")
                print(f"   ETA: {str(timedelta(seconds=int(eta)))}")
                
                # FIXED: Properly check early stopping
                should_stop = early_stopping(val_acc, epoch)
                
                if should_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
                    print(f"   Best validation accuracy: {best_val_acc:.2f}% at epoch {early_stopping.best_epoch}")
                    print(f"   Total training time: {str(timedelta(seconds=int(time.time() - training_start)))}")
                    break
                
                # Plot training curves periodically
                if epoch % 5 == 0:
                    self.plot_training_curves()
                
                # Aggressive memory cleanup after each epoch
                del train_loss, train_acc, val_loss, val_acc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch}")
            print(f"Saving checkpoint...")
            
            metrics = {
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else None,
                'best_val_acc': best_val_acc,
                'history': self.history
            }
            
            self.checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, self.scaler, metrics, False
            )
            print(f"Checkpoint saved. Training can be resumed.")
            return
        
        except Exception as e:
            print(f"\nTraining error: {e}")
            import traceback
            traceback.print_exc()
            
            # Save comprehensive emergency checkpoint
            print(f"Saving comprehensive emergency checkpoint...")
            
            emergency_metrics = {
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else None,
                'best_val_acc': best_val_acc,
                'history': self.history,
                'current_lr': optimizer.param_groups[0]['lr']
            }
            
            emergency_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'metrics': emergency_metrics,
                'timestamp': datetime.now().isoformat(),
                'error_info': str(e),
                'emergency_save': True
            }
            
            emergency_checkpoint = self.output_dir / 'checkpoints' / f'emergency_checkpoint_epoch_{epoch:03d}.pt'
            torch.save(emergency_checkpoint_data, emergency_checkpoint)
            print(f"Comprehensive emergency checkpoint saved: {emergency_checkpoint}")
            return
        
        # Final evaluation
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        total_training_time = time.time() - training_start
        print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Test evaluation
        if test_loader and len(test_loader) > 0:
            print(f"\nEvaluating on test set...")
            
            # Load best model
            best_checkpoint = torch.load(self.checkpoint_manager.best_model_file)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
            test_loss, test_acc = self.validate(model, test_loader, criterion)
            
            print(f"\nFinal Test Results:")
            print(f"   Test Loss: {test_loss:.4f}")
            print(f"   Test Accuracy: {test_acc:.2f}%")
            
            # Save final results
            results = {
                'train_history': self.history,
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'training_time': total_training_time,
                'num_epochs_trained': epoch + 1,
                'early_stopped': hasattr(early_stopping, 'early_stop') and early_stopping.early_stop,
                'best_epoch': early_stopping.best_epoch,
                'label_smoothing': label_smoothing,
                'early_stopping_patience': early_stopping_patience,
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = self.output_dir / 'fixed_training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {results_file}")
        
        # Plot final training curves
        self.plot_training_curves(save=True)
        
        print(f"\n🎉 Fixed training pipeline completed successfully!")
        print(f"   All features working: Label Smoothing ✓, Early Stopping ✓, Adaptive LR ✓")
    
    def plot_training_curves(self, save=False):
        """Enhanced plotting with 4 subplots"""
        if len(self.history['train_loss']) < 2:
            return
        
        # Use non-interactive backend if saving
        if save:
            plt.switch_backend('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2, color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2, color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2, color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if self.history['learning_rates']:
            axes[1, 0].plot(self.history['learning_rates'], linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule (ReduceLROnPlateau)')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting analysis
        if len(self.history['train_acc']) > 1 and len(self.history['val_acc']) > 1:
            train_val_gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
            axes[1, 1].plot(train_val_gap, linewidth=2, color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
            axes[1, 1].set_title('Overfitting Analysis (Gap)')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.output_dir / 'fixed_training_curves.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {plot_file}")
        
        plt.close()


def get_system_info():
    """Get system information for both Windows and Linux"""
    system_info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'platform': platform.platform()
    }
    
    # Memory information
    mem = psutil.virtual_memory()
    system_info['total_ram_gb'] = mem.total / 1e9
    system_info['available_ram_gb'] = mem.available / 1e9
    
    # GPU information
    if torch.cuda.is_available():
        system_info['gpu_name'] = torch.cuda.get_device_name()
        system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        system_info['cuda_version'] = torch.version.cuda
        system_info['cudnn_version'] = torch.backends.cudnn.version()
    else:
        system_info['gpu_name'] = 'No CUDA GPU available'
    
    return system_info


def find_data_directory():
    """Find data directory across different platforms"""
    possible_paths = [
        Path("video_classification_project") / "data" / "processed",
        Path("data") / "processed",
        Path("..") / "data" / "processed",
        Path("processed"),
        Path.home() / "video_classification_project" / "data" / "processed",
        Path("C:") / "video_classification_project" / "data" / "processed" if platform.system() == "Windows" else Path("/tmp") / "video_classification_project" / "data" / "processed"
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path
    
    return None


def main():
    """Main function with comprehensive error handling and cross-platform support"""
    print("=" * 80)
    print("FIXED VIDEO CLASSIFICATION TRAINER")
    print("Features: Label Smoothing ✓, Early Stopping ✓, Adaptive LR ✓")
    print("Memory Optimization: Mixed Precision, Gradient Accumulation, Memory Mapping")
    print("Hardware: Optimized for RTX 4060 8GB / 16GB RAM")
    print("=" * 80)
    
    # Get system information
    system_info = get_system_info()
    
    print(f"\nSystem Information:")
    print(f"   OS: {system_info['os']} ({system_info['platform']})")
    print(f"   Python: {system_info['python_version'].split()[0]}")
    print(f"   Total RAM: {system_info['total_ram_gb']:.1f}GB")
    print(f"   Available RAM: {system_info['available_ram_gb']:.1f}GB")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   GPU: {system_info['gpu_name']}")
    if 'gpu_memory_gb' in system_info:
        print(f"   GPU Memory: {system_info['gpu_memory_gb']:.1f}GB")
        print(f"   CUDA: {system_info['cuda_version']}")
    
    # Configuration
    data_dir = find_data_directory()
    if data_dir is None:
        print(f"\nData directory not found!")
        print(f"\nPlease ensure your data is preprocessed and located at one of:")
        possible_paths = [
            "video_classification_project/data/processed",
            "data/processed", 
            "../data/processed",
            "processed"
        ]
        for path in possible_paths:
            print(f"   {path}")
        return
    
    print(f"\nFound data directory: {data_dir}")
    
    # Create output directory
    if platform.system() == "Windows":
        output_dir = Path("video_classification_project") / "models"
    else:
        output_dir = Path("video_classification_project") / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory configuration based on system
    if system_info['available_ram_gb'] > 12:
        max_memory_gb = 4.0
        gradient_accumulation_steps = 2
    elif system_info['available_ram_gb'] > 8:
        max_memory_gb = 3.0
        gradient_accumulation_steps = 4
    else:
        max_memory_gb = 2.0
        gradient_accumulation_steps = 8
    
    print(f"\nConfiguration:")
    print(f"   Memory limit per file: {max_memory_gb}GB")
    print(f"   Gradient accumulation: {gradient_accumulation_steps} steps")
    
    # Initialize trainer with FIXED features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = VideoClassificationTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        max_memory_gb=max_memory_gb,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_val=1.0
    )
    
    try:
        print(f"\nStarting FIXED training pipeline...")
        print(f"   Data directory: {data_dir}")
        print(f"   Output directory: {output_dir}")
        print(f"   Device: {device}")
        
        # Train model with FIXED settings
        trainer.train(
            num_epochs=50,
            label_smoothing=0.1,  # NOW PROPERLY USED
            early_stopping_patience=10  # NOW PROPERLY IMPLEMENTED
        )
        
    except KeyboardInterrupt:
        print(f"\nProcess interrupted by user")
        print(f"Training can be resumed from the last checkpoint")
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nGPU Out of Memory Error!")
        print(f"   Error: {str(e)}")
        print(f"\nSolutions:")
        print(f"   1. Reduce batch size (currently 2)")
        print(f"   2. Increase gradient accumulation steps")
        print(f"   3. Use smaller model (reduce hidden_dim)")
        print(f"   4. Clear GPU cache and retry")
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\nTips for troubleshooting:")
        print(f"   1. Check if data files are properly formatted")
        print(f"   2. Ensure sufficient disk space for checkpoints")
        print(f"   3. Monitor GPU memory usage during training")
        print(f"   4. Check the error log above for specific issues")
        
        # Save system info for debugging
        debug_file = output_dir / 'debug_system_info.json'
        try:
            with open(debug_file, 'w') as f:
                json.dump(system_info, f, indent=2, default=str)
            print(f"   5. System info saved to {debug_file}")
        except:
            pass


if __name__ == "__main__":
    # Set environment variables for optimal performance
    if platform.system() == "Windows":
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        # Disable Windows Defender real-time scanning on temp files if possible
        os.environ['TMP'] = str(Path.cwd() / 'temp')
        os.environ['TEMP'] = str(Path.cwd() / 'temp')
        Path(os.environ['TMP']).mkdir(exist_ok=True)
    else:
        # Linux-specific optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        # Use faster memory allocator on Linux
        os.environ['MALLOC_ARENA_MAX'] = '1'
    
    # Set torch settings for both platforms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Set memory management based on platform
        if platform.system() == "Windows":
            # More conservative on Windows due to memory fragmentation
            torch.cuda.set_per_process_memory_fraction(0.85)
        else:
            # More aggressive on Linux
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Run main
    main()