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
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
import warnings
import pickle
warnings.filterwarnings('ignore')


class VideoDataset(Dataset):
    """Custom dataset for loading preprocessed videos with detailed progress tracking"""
    def __init__(self, data_dir, split='train'):
        print(f"\n{'='*60}")
        print(f"INITIALIZING {split.upper()} DATASET")
        print(f"{'='*60}")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.videos = []
        self.labels = []
        self.filenames = []
        self.category_mapping = {}
        
        print(f"Searching for data in: {self.data_dir}")
        print(f"Expected structure: {split}/<Category>/<Sub-Category>/processed_data.pt")
        
        # First, scan for all available data files
        data_files = []
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            print(f"❌ Split directory not found: {split_dir}")
            print("Available directories:")
            if self.data_dir.exists():
                for item in self.data_dir.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
            return
        
        print(f"✅ Found split directory: {split_dir}")
        print(f"Scanning for data files...")
        
        # Scan directory structure with progress
        category_dirs = list(split_dir.glob("*"))
        category_dirs = [d for d in category_dirs if d.is_dir()]
        
        if not category_dirs:
            print(f"❌ No category directories found in {split_dir}")
            return
        
        print(f"Found {len(category_dirs)} category directories:")
        for cat_dir in category_dirs:
            print(f"  - {cat_dir.name}")
        
        # Collect all data files
        for category_dir in tqdm(category_dirs, desc="Scanning categories", leave=False):
            subcat_dirs = list(category_dir.glob("*"))
            subcat_dirs = [d for d in subcat_dirs if d.is_dir()]
            
            print(f"\n📁 Category: {category_dir.name}")
            print(f"   Found {len(subcat_dirs)} subcategories")
            
            for subcat_dir in subcat_dirs:
                # Look for processed_data.pt first
                data_file = subcat_dir / 'processed_data.pt'
                if data_file.exists():
                    data_files.append(data_file)
                    print(f"   ✅ {subcat_dir.name}/processed_data.pt ({data_file.stat().st_size / 1e6:.1f}MB)")
                else:
                    # Look for individual .pt files
                    pt_files = list(subcat_dir.glob("*.pt"))
                    if pt_files:
                        data_files.extend(pt_files)
                        total_size = sum(f.stat().st_size for f in pt_files) / 1e6
                        print(f"   ⚠️  {subcat_dir.name}: {len(pt_files)} individual .pt files ({total_size:.1f}MB)")
                    else:
                        print(f"   ❌ {subcat_dir.name}: No .pt files found")
        
        if not data_files:
            print(f"\n❌ No data files found in {split} split!")
            print("Make sure your data is preprocessed and saved as .pt files.")
            return
        
        print(f"\n🔍 Found {len(data_files)} data files to load")
        total_size = sum(f.stat().st_size for f in data_files) / 1e6
        print(f"📊 Total data size: {total_size:.1f}MB")
        print(f"⏳ Starting data loading...")
        
        # Load all data files with detailed progress
        failed_files = []
        loaded_files = 0
        
        progress_bar = tqdm(data_files, desc="Loading data files", unit="file")
        
        for data_file in progress_bar:
            try:
                # Update progress bar with current file info
                file_size = data_file.stat().st_size / 1e6
                progress_bar.set_postfix({
                    'file': data_file.name[:20] + '...' if len(data_file.name) > 20 else data_file.name,
                    'size': f'{file_size:.1f}MB',
                    'loaded': loaded_files
                })
                
                # Load the data
                start_time = time.time()
                data = torch.load(data_file, map_location='cpu')
                load_time = time.time() - start_time
                
                # Handle different data formats
                if isinstance(data, dict):
                    if 'videos' in data and 'labels' in data:
                        # Batch format
                        self.videos.append(data['videos'])
                        self.labels.append(data['labels'])
                        if 'filenames' in data:
                            self.filenames.extend(data['filenames'])
                        if 'category_mapping' in data:
                            for cat, idx in data['category_mapping'].items():
                                self.category_mapping[cat] = idx
                        
                        progress_bar.set_postfix({
                            'file': data_file.name[:15] + '...',
                            'size': f'{file_size:.1f}MB',
                            'videos': len(data['videos']),
                            'time': f'{load_time:.2f}s'
                        })
                    else:
                        print(f"\n⚠️  Unexpected data format in {data_file}")
                        failed_files.append(data_file)
                        continue
                else:
                    # Individual tensor files
                    print(f"\n⚠️  Individual tensor format not fully supported: {data_file}")
                    failed_files.append(data_file)
                    continue
                
                loaded_files += 1
                
            except Exception as e:
                print(f"\n❌ Failed to load {data_file}: {str(e)}")
                failed_files.append(data_file)
                continue
        
        progress_bar.close()
        
        # Consolidate loaded data
        if self.videos:
            print(f"\n⚙️  Consolidating {len(self.videos)} batches...")
            
            # Show memory usage before concatenation
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1e9
                print(f"📈 GPU memory before concatenation: {gpu_memory_before:.2f}GB")
            
            consolidation_start = time.time()
            
            try:
                self.videos = torch.cat(self.videos, dim=0)
                self.labels = torch.cat(self.labels, dim=0)
                
                consolidation_time = time.time() - consolidation_start
                print(f"⏱️  Consolidation time: {consolidation_time:.2f}s")
                
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1e9
                    print(f"📈 GPU memory after consolidation: {gpu_memory_after:.2f}GB")
                
            except Exception as e:
                print(f"❌ Failed to consolidate data: {str(e)}")
                print("This might be due to insufficient memory or incompatible tensor shapes")
                return
            
            # Final validation and summary
            print(f"\n{'='*60}")
            print(f"DATASET LOADING COMPLETED")
            print(f"{'='*60}")
            print(f"✅ Successfully loaded: {loaded_files}/{len(data_files)} files")
            if failed_files:
                print(f"❌ Failed to load: {len(failed_files)} files")
            
            print(f"\n📊 DATASET SUMMARY:")
            print(f"   Total videos: {len(self.videos):,}")
            print(f"   Video shape: {self.videos.shape}")
            print(f"   Labels shape: {self.labels.shape}")
            print(f"   Memory usage: {self.videos.numel() * 4 / 1e9:.2f}GB")
            print(f"   Categories: {len(self.category_mapping)}")
            
            if self.category_mapping:
                print(f"\n🏷️  CATEGORY MAPPING:")
                for cat, idx in sorted(self.category_mapping.items(), key=lambda x: x[1]):
                    count = (self.labels == idx).sum().item()
                    print(f"   {idx}: {cat} ({count:,} samples)")
            
            # Data validation
            print(f"\n🔍 DATA VALIDATION:")
            if len(self.videos) != len(self.labels):
                print(f"❌ Mismatch: {len(self.videos)} videos but {len(self.labels)} labels")
                raise ValueError(f"Data mismatch: videos and labels have different lengths")
            else:
                print(f"✅ Videos and labels match: {len(self.videos)} samples")
            
            unique_labels = set(self.labels.tolist())
            expected_labels = len(self.category_mapping)
            if len(unique_labels) != expected_labels:
                print(f"⚠️  Expected {expected_labels} unique labels, found {len(unique_labels)}")
                print(f"   Found labels: {sorted(unique_labels)}")
                print(f"   Expected labels: {sorted(self.category_mapping.values())}")
            else:
                print(f"✅ Label validation passed: {len(unique_labels)} categories")
            
            # Memory and performance info
            tensor_memory = self.videos.numel() * self.videos.element_size() / 1e9
            print(f"\n💾 MEMORY USAGE:")
            print(f"   Videos tensor: {tensor_memory:.2f}GB")
            print(f"   Est. batch memory (batch=4): {tensor_memory * 4 / len(self.videos):.2f}GB")
            
            if failed_files:
                print(f"\n❌ FAILED FILES ({len(failed_files)}):")
                for failed_file in failed_files[:10]:  # Show first 10
                    print(f"   - {failed_file}")
                if len(failed_files) > 10:
                    print(f"   ... and {len(failed_files) - 10} more")
        
        else:
            print(f"\n❌ NO DATA LOADED!")
            print("Possible issues:")
            print("1. Data files are corrupted or in wrong format")
            print("2. Insufficient memory to load data")
            print("3. Data directory structure is incorrect")
            print(f"\nExpected structure:")
            print(f"  {self.data_dir}/")
            print(f"  ├── {split}/")
            print(f"  │   ├── Category1/")
            print(f"  │   │   └── SubCategory1/")
            print(f"  │   │       └── processed_data.pt")
            print(f"  │   └── Category2/")
            print(f"  │       └── SubCategory2/")
            print(f"  │           └── processed_data.pt")
    
    def __len__(self):
        return len(self.videos) if hasattr(self, 'videos') and len(self.videos) > 0 else 0
    
    def __getitem__(self, idx):
        if len(self.videos) == 0:
            raise RuntimeError("Dataset is empty. Check data loading logs above.")
        return self.videos[idx], self.labels[idx]


class EnhancedCNNLSTM(nn.Module):
    """Enhanced CNN-LSTM architecture with attention mechanism for maximum accuracy"""
    def __init__(self, num_classes=4, hidden_dim=1024, num_layers=4, dropout=0.2, 
                 backbone='efficientnet_b4', bidirectional=True, attention=True):
        super(EnhancedCNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = attention
        
        # Select CNN backbone - EfficientNet-B4 for best accuracy
        if backbone == 'efficientnet_b4':
            self.cnn = models.efficientnet_b4(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
        elif backbone == 'densenet161':
            self.cnn = models.densenet161(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Identity()
        elif backbone == 'resnet101':
            self.cnn = models.resnet101(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        else:  # resnet50 fallback
            self.cnn = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        
        # Feature projection with residual connection
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-layer LSTM with higher capacity
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Multi-head attention mechanism
        if self.use_attention:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        
        # Enhanced classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process each frame through CNN
        x = x.view(-1, channels, height, width)
        features = self.cnn(x)
        features = features.view(batch_size * num_frames, -1)
        features = self.feature_projection(features)
        features = features.view(batch_size, num_frames, -1)
        
        # Process temporal features through LSTM
        lstm_out, _ = self.lstm(features)
        
        if self.use_attention:
            # Apply multi-head self-attention
            attended_features, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            attended_features = self.attention_norm(attended_features + lstm_out)  # Residual connection
            # Global average pooling over sequence
            attended_features = attended_features.mean(dim=1)
        else:
            # Use last hidden state
            attended_features = lstm_out[:, -1, :]
        
        # Final classification
        output = self.classifier(attended_features)
        return output


class VideoClassificationTrainer:
    def __init__(self, data_dir, output_dir, device='cuda'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Auto-detect GPU and enable optimizations
        self._setup_device_optimizations()
        
    def _setup_device_optimizations(self):
        """Auto-detect GPU and enable optimizations"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n🚀 GPU DETECTED")
            print(f"   Device: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Enable mixed precision for all modern GPUs
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"   ✅ Mixed precision enabled")
            
            # Enable optimized attention for Ampere+ GPUs (RTX 30/40 series)
            if any(x in gpu_name for x in ["RTX 40", "RTX 30", "A100", "RTX 4060"]):
                torch.backends.cuda.enable_flash_sdp(True)
                print(f"   ✅ Flash Attention enabled")
        else:
            print(f"\n⚠️  Using CPU - training will be significantly slower")
            self.scaler = None
    
    def create_data_loaders(self):
        """Create optimized data loaders with progress tracking"""
        print(f"\n{'='*60}")
        print(f"CREATING DATA LOADERS")
        print(f"{'='*60}")
        
        # Create datasets with progress tracking
        train_dataset = VideoDataset(self.data_dir, split='train')
        val_dataset = VideoDataset(self.data_dir, split='val')
        test_dataset = VideoDataset(self.data_dir, split='test')
        
        # Check if datasets loaded successfully
        datasets_info = [
            ('Train', train_dataset, len(train_dataset)),
            ('Validation', val_dataset, len(val_dataset)),
            ('Test', test_dataset, len(test_dataset))
        ]
        
        print(f"\n📊 DATASET SUMMARY:")
        total_samples = 0
        for name, dataset, size in datasets_info:
            if size > 0:
                print(f"   ✅ {name}: {size:,} samples")
                total_samples += size
            else:
                print(f"   ❌ {name}: No samples loaded!")
        
        print(f"   📈 Total: {total_samples:,} samples")
        
        if total_samples == 0:
            raise RuntimeError("No data loaded! Please check your data directory and preprocessing.")
        
        # Optimized DataLoader settings
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Adaptive batch size based on GPU memory
            if gpu_memory >= 12:  # RTX 4070 Ti and above
                batch_size = 8
            elif gpu_memory >= 8:  # RTX 4060 Ti, RTX 3070
                batch_size = 4
            else:  # RTX 4060, RTX 3060
                batch_size = 2
            
            num_workers = min(8, os.cpu_count() // 2)
            pin_memory = True
        else:
            batch_size = 1
            num_workers = 0
            pin_memory = False
        
        print(f"\n⚙️  DATALOADER CONFIGURATION:")
        print(f"   Batch size: {batch_size}")
        print(f"   Workers: {num_workers}")
        print(f"   Pin memory: {pin_memory}")
        
        # Create DataLoaders
        dataloaders = []
        for name, dataset, _ in datasets_info:
            if len(dataset) > 0:
                is_train = (name == 'Train')
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=is_train,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=is_train,
                    persistent_workers=num_workers > 0
                )
                dataloaders.append(loader)
                print(f"   ✅ {name}: {len(loader)} batches")
            else:
                # Create empty loader for consistency
                dataloaders.append(None)
                print(f"   ❌ {name}: No data")
        
        return dataloaders[0], dataloaders[1], dataloaders[2]


def main():
    """Main function with enhanced error handling and progress tracking"""
    print("=" * 80)
    print("VIDEO CLASSIFICATION - MAXIMUM ACCURACY MODE")
    print("Enhanced with Comprehensive Progress Tracking")
    print("=" * 80)
    
    # Setup paths
    data_dir = Path("video_classification_project/data/processed")
    output_dir = Path("video_classification_project/models")
    
    # Verify data directory exists
    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory not found!")
        print(f"   Expected: {data_dir}")
        print(f"   Current working directory: {Path.cwd()}")
        
        # Look for alternative locations
        alternative_paths = [
            Path("data/processed"),
            Path("../data/processed"),
            Path("./processed"),
        ]
        
        print(f"\n🔍 Searching for alternative data locations...")
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"   ✅ Found: {alt_path}")
                data_dir = alt_path
                break
            else:
                print(f"   ❌ Not found: {alt_path}")
        else:
            print(f"\n💡 SOLUTION:")
            print(f"   1. Make sure your data is preprocessed")
            print(f"   2. Check the directory structure:")
            print(f"      data/processed/")
            print(f"      ├── train/")
            print(f"      ├── val/")
            print(f"      └── test/")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = VideoClassificationTrainer(data_dir, output_dir, device)
    
    try:
        # Test data loading first
        print(f"\n🧪 TESTING DATA LOADING...")
        train_loader, val_loader, test_loader = trainer.create_data_loaders()
        
        if train_loader is None or len(train_loader) == 0:
            print(f"❌ No training data available. Cannot proceed.")
            return
        
        print(f"\n✅ Data loading successful!")
        print(f"   Ready to start training...")
        
        # You would continue with model setup and training here
        # trainer.setup_ensemble_models()
        # trainer.train()
        
    except KeyboardInterrupt:
        print(f"\n⛔ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set optimal torch settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable tensor core usage
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    main()