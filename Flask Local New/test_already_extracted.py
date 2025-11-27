"""
SINGLE VIDEO INFERENCE SCRIPT - Using Pre-Extracted Features OR External Video
Extended to support downloading / extracting features from arbitrary video URLs/local files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
import shutil
import subprocess
import tempfile
import os
import math
import cv2
import requests
from urllib.parse import urlparse
import torchvision.transforms as T
import torchvision.models as models

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / 'src' / 'data'

if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

# Import model architecture
try:
    from model_train_new import SuperEnhancedTemporalModel
    print("✓ Successfully imported model architecture")
except (ImportError, ModuleNotFoundError) as e:
    print(f"✗ Error importing model: {e}")
    sys.exit(1)


# ------------------------
# Helpers: download + feature extraction for arbitrary video
# ------------------------
def download_video_if_needed(url_or_path, tmp_dir=None):
    """
    If url_or_path is a local file -> return it.
    If it's a URL:
      - try yt_dlp to download the best MP4
      - fallback: try requests to download raw bytes (only works for direct .mp4 links)
    Returns local filepath.
    """
    # Local file?
    if os.path.exists(url_or_path):
        return url_or_path

    parsed = urlparse(url_or_path)
    if not parsed.scheme:
        raise ValueError(f"Not a valid file or URL: {url_or_path}")

    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="download_video_")
    out_path = os.path.join(tmp_dir, "downloaded_video.mp4")

    # Try yt_dlp if available
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': out_path,
            'quiet': True,
            'merge_output_format': 'mp4',
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        # print warning and fallback
        print(f"⚠ yt_dlp failed or not available: {e}")

    # Fallback for direct links
    try:
        print("Attempting direct download (requests)...")
        r = requests.get(url_or_path, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        raise RuntimeError(f"Could not download video: {e}")

    raise RuntimeError("Download attempt failed.")


def extract_video_features(video_path, model_feature_dim, num_frames=73, device='cpu'):
    """
    EfficientNet-B0 backbone extractor (matches 1280 feature_dim).
    - Samples num_frames frames uniformly across the video.
    - Uses torchvision.models.efficientnet_b0 pretrained and returns [T, 1280].
    - device may be 'cpu', 'cuda' or a torch.device.
    """
    import torch

    # normalize device
    if isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device if torch.cuda.is_available() and device in ('cuda', 'cuda:0') else 'cpu')

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        # fallback: read/count frames
        frames_tmp = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_tmp.append(frame)
        total_frames = len(frames_tmp)
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if total_frames == 0:
            raise RuntimeError("Video contains no frames.")

    # Compute indices to sample uniformly
    if total_frames <= num_frames:
        indices = list(range(total_frames)) + [max(total_frames - 1, 0)] * (num_frames - total_frames)
    else:
        indices = [math.floor(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    # Load EfficientNet-B0 backbone (pretrained)
    try:
        eff = models.efficientnet_b0(pretrained=True)
    except Exception as e:
        raise RuntimeError("efficientnet_b0 not available in torchvision; install a newer torchvision or ask for a timm variant.") from e

    eff.eval()
    # remove classifier: children() up to last pooling layer; for efficientnet, use features + avgpool (features -> [N,1280,7,7], then adaptive pool happens in classifier)
    # We'll use eff.features + global pooling to get 1280-d vector
    backbone = torch.nn.Sequential(*list(eff.features.children())).to(dev)
    backbone.eval()

    # Preprocessing: same as EfficientNet defaults (224)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    feats = []
    target_positions = set(indices)
    read_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx in target_positions:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = transform(frame_rgb)
            feats.append(inp)
            if len(feats) >= num_frames:
                break
        read_idx += 1

    cap.release()

    if len(feats) < num_frames:
        if len(feats) == 0:
            raise RuntimeError("No frames extracted.")
        last = feats[-1]
        while len(feats) < num_frames:
            feats.append(last)

    batch = torch.stack(feats, dim=0).to(dev)  # [T,3,H,W]

    with torch.no_grad():
        # run through feature extractor
        x = backbone(batch)  # [T, C, H, W], C should be 1280 for effnet_b0
        # global avg pool to get [T, C]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # [T, 1280]

    D_backbone = x.size(1)
    if D_backbone != model_feature_dim:
        # if dims still mismatch, project deterministically via linear (but ideally this shouldn't happen)
        proj = torch.nn.Linear(D_backbone, model_feature_dim).to(dev)
        with torch.no_grad():
            x = proj(x)

    return x.cpu()



# ------------------------
# Original classes (unchanged), with small fix for class_names ordering
# ------------------------

class SingleVideoFeatureLoader:
    """Load pre-extracted features for a single video from .h5 file"""
    
    def __init__(self, features_dir, processed_test_dir):
        self.features_dir = Path(features_dir)
        self.processed_test_dir = Path(processed_test_dir)  # data/processed/test
        
        print(f"\n{'='*70}")
        print("INITIALIZING FEATURE LOADER")
        print(f"{'='*70}")
        print(f"Features dir: {self.features_dir}")
        print(f"Processed test dir: {self.processed_test_dir}")
        
        # Load test features .h5 file
        self.h5_path = None
        h5_files = list(self.features_dir.glob('test_features*.h5'))
        
        if not h5_files:
            print(f"✗ No test_features*.h5 file found in {self.features_dir}")
            sys.exit(1)
        
        self.h5_path = h5_files[0]
        print(f"Found features file: {self.h5_path.name}")
        
        # Load category mapping
        with h5py.File(self.h5_path, 'r') as f:
            self.category_mapping = json.loads(f.attrs['category_mapping'])
            self.num_samples = len(f['features'])
            print(f"Total samples in test set: {self.num_samples}")
        
        # Create reverse mapping
        self.idx_to_category = {idx: name for name, idx in self.category_mapping.items()}
        self.class_names = [self.idx_to_category[i] for i in range(len(self.category_mapping))]
        
        print(f"Classes: {self.class_names}")
        
        # Build video name index
        self._build_video_index()
        
        print(f"✓ Feature loader ready\n")
    
    def _build_video_index(self):
        """Build an index mapping video names to their position in .h5 file"""
        print("\nBuilding video index from processed_data.pt files...")
        
        self.video_index = {}  # video_name -> (h5_index, label, category_name)
        
        with h5py.File(self.h5_path, 'r') as f:
            labels = f['labels'][:]
        
        # The .h5 file was built from data/processed/test structure
        # We need to scan processed_data.pt files in the same order
        
        h5_idx = 0
        
        for category_name in sorted(self.category_mapping.keys()):
            category_dir = self.processed_test_dir / category_name
            
            if not category_dir.exists():
                print(f"   ⊘ Category directory not found: {category_name}")
                continue
            
            print(f"   Scanning category: {category_name}")
            
            # Get all subcategories in sorted order
            for subcat_dir in sorted(category_dir.iterdir()):
                if not subcat_dir.is_dir():
                    continue
                
                processed_file = subcat_dir / 'processed_data.pt'
                if not processed_file.exists():
                    continue
                
                try:
                    # Load filenames from processed_data.pt
                    data = torch.load(processed_file, map_location='cpu')
                    filenames = data.get('filenames', [])
                    
                    print(f"      - {subcat_dir.name}: {len(filenames)} videos")
                    
                    # Map each filename to h5 index
                    for video_name in filenames:
                        if h5_idx < len(labels):
                            label = labels[h5_idx]
                            self.video_index[video_name] = (h5_idx, label, category_name)
                            h5_idx += 1
                        else:
                            print(f"        Warning: h5_idx {h5_idx} exceeds labels length")
                            break
                    
                except Exception as e:
                    print(f"      ✗ Error loading {subcat_dir.name}: {e}")
                    continue
        
        print(f"\n✓ Indexed {len(self.video_index)} videos")
        
        # Debug: show some indexed videos
        if self.video_index:
            print(f"\nFirst 5 indexed videos:")
            for i, (video_name, (h5_idx, label, cat)) in enumerate(list(self.video_index.items())[:5]):
                print(f"   {i+1}. {video_name} -> h5_idx={h5_idx}, cat={cat}")
    
    def list_available_videos(self):
        """List all videos that can be loaded"""
        print(f"\n{'='*70}")
        print(f"AVAILABLE VIDEOS IN TEST SET")
        print(f"{'='*70}\n")
        
        videos_by_category = {}
        
        for video_name, (h5_idx, label, category_name) in self.video_index.items():
            if category_name not in videos_by_category:
                videos_by_category[category_name] = []
            videos_by_category[category_name].append(video_name)
        
        total = sum(len(v) for v in videos_by_category.values())
        print(f"Total videos: {total}\n")
        
        all_videos = []
        for category_name in sorted(videos_by_category.keys()):
            videos = sorted(videos_by_category[category_name])
            print(f"\n{category_name} ({len(videos)} videos):")
            print("-" * 60)
            
            for i, video_name in enumerate(videos[:20], 1):  # Show first 20
                global_idx = len(all_videos) + 1
                print(f"   {global_idx:3d}. {video_name}")
                all_videos.append(video_name)
            
            if len(videos) > 20:
                print(f"   ... and {len(videos) - 20} more")
                # Add remaining videos to all_videos list
                for video_name in videos[20:]:
                    all_videos.append(video_name)
        
        print("\n" + "="*70 + "\n")
        
        return all_videos, videos_by_category
    
    def load_video_features(self, video_name):
        """Load pre-extracted features for a specific video"""
        print(f"\nLoading features for: {video_name}")
        
        # Check if video is in index
        if video_name not in self.video_index:
            print(f"✗ Video not found in index")
            return None, None, None
        
        h5_idx, label, category_name = self.video_index[video_name]
        
        print(f"Video index: {h5_idx}")
        print(f"Category: {category_name}")
        print(f"Label: {label}")
        
        # Load features from .h5
        with h5py.File(self.h5_path, 'r') as f:
            num_frames_val = f['num_frames'][h5_idx]
            features = f['features'][h5_idx][:num_frames_val]  # [T, feature_dim]
        
        features = torch.from_numpy(features.copy())
        
        print(f"Features shape: {features.shape}")
        print(f"✓ Loaded features\n")
        
        return features, label, category_name


class SingleVideoClassifier:
    """Classify single video using pre-extracted features - same pipeline as testing script"""
    
    def __init__(self, checkpoint_paths, device='cuda'):
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"SINGLE VIDEO CLASSIFIER")
        print(f"{'='*70}")
        print(f"Models: {len(self.checkpoint_paths)}")
        for i, path in enumerate(self.checkpoint_paths, 1):
            print(f"   {i}. {path.name}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        self.models = []
        self.class_names = None
        self.num_classes = None
        self.feature_dim = None
    
    def load_models(self):
        """Load all trained models"""
        print(f"Loading {len(self.checkpoint_paths)} model(s)...\n")
        
        for i, checkpoint_path in enumerate(self.checkpoint_paths, 1):
            print(f"   Loading model {i}/{len(self.checkpoint_paths)}: {checkpoint_path.name}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get model configuration (same as testing script)
            config = checkpoint.get('model_config', checkpoint.get('config', {}))
            
            feature_dim = config.get('feature_dim', 1280)
            hidden_dim = config.get('hidden_dim', 768)
            num_classes = config.get('num_classes', 4)
            num_lstm_layers = config.get('num_lstm_layers', 4)
            num_attention_heads = config.get('num_attention_heads', 12)
            dropout = config.get('dropout', 0.4)
            bidirectional = config.get('bidirectional', True)
            
            # Store for first model
            if i == 1:
                self.feature_dim = feature_dim
                self.num_classes = num_classes
                # IMPORTANT: set class_names from dataset order (this should match your h5 mapping)
                self.class_names = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']

            
            # Initialize model (same as testing script)
            model = SuperEnhancedTemporalModel(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_lstm_layers=num_lstm_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                bidirectional=bidirectional
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            
            # Display info
            if 'best_val_acc' in checkpoint:
                print(f"      Val accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        print(f"\n✓ All {len(self.models)} model(s) loaded successfully")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Classes: {self.class_names}\n")
    
    def predict_standard(self, features):
        """Standard prediction (no TTA) - same as test_standard()"""
        all_model_predictions = []
        
        with torch.no_grad():
            features_batch = features.unsqueeze(0).to(self.device)  # [1, T, D]
            lengths = torch.tensor([features.shape[0]], device=self.device)
            
            for model in self.models:
                outputs = model(features_batch, lengths)  # [1, num_classes]
                probs = F.softmax(outputs, dim=1)
                all_model_predictions.append(probs.squeeze(0).cpu())
        
        # Ensemble: average predictions
        ensemble_probs = torch.stack(all_model_predictions).mean(dim=0)
        
        return ensemble_probs
    
    def predict_with_tta(self, features):
        """Prediction with TTA - same as test_with_tta()"""
        tta_predictions = []
        
        # TTA Mode 1: Original
        probs = self.predict_standard(features)
        tta_predictions.append(probs)
        
        # TTA Mode 2: Reverse
        features_reversed = torch.flip(features, dims=[0])
        probs = self.predict_standard(features_reversed)
        tta_predictions.append(probs)
        
        # TTA Mode 3: Speed up (skip frames)
        if features.shape[0] > 10:
            indices = torch.linspace(0, features.shape[0]-1, features.shape[0]//2).long()
            features_speedup = features[indices]
            probs = self.predict_standard(features_speedup)
            tta_predictions.append(probs)
        
        # TTA Mode 4: Speed down (interpolate frames)
        if features.shape[0] > 10:
            indices = torch.linspace(0, features.shape[0]-1, int(features.shape[0]*1.5)).long()
            indices = indices.clamp(max=features.shape[0]-1)
            features_speeddown = features[indices]
            probs = self.predict_standard(features_speeddown)
            tta_predictions.append(probs)
        
        # Average TTA predictions
        ensemble_probs = torch.stack(tta_predictions).mean(dim=0)
        
        return ensemble_probs
    
    def classify_video(self, features, true_label=None, video_name="Unknown", use_tta=False):
        """Classify a single video - follows same pipeline as testing script"""
        
        print(f"\n{'='*70}")
        print(f"CLASSIFYING VIDEO")
        print(f"{'='*70}")
        print(f"Video: {video_name}")
        if true_label is not None:
            true_class_name = self.class_names[true_label]
            print(f"True label: {true_label} -> {true_class_name}")
        print(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
        print(f"Ensemble size: {len(self.models)}")
        print(f"{'='*70}\n")
        
        # Make prediction
        print(f"Running inference...")
        
        if use_tta:
            print(f"Using TTA with 4 augmentation modes...")
            probs = self.predict_with_tta(features)
        else:
            print(f"Using standard inference...")
            probs = self.predict_standard(features)
        
        # Get results
        confidence_scores = probs.numpy() * 100
        predicted_class_idx = confidence_scores.argmax()
        predicted_class = self.class_names[predicted_class_idx]
        predicted_confidence = confidence_scores[predicted_class_idx]
        
        # Check if correct
        is_correct = (predicted_class_idx == true_label) if true_label is not None else None
        
        # Display results
        print(f"\n{'='*70}")
        print("CLASSIFICATION RESULTS")
        print(f"{'='*70}\n")
        
        print(f"Video: {video_name}")
        if true_label is not None:
            true_class_name = self.class_names[true_label]
            print(f"True Class: {true_class_name}")
        print(f"\n🎯 Predicted Class: {predicted_class}")
        print(f"📊 Confidence: {predicted_confidence:.2f}%")
        
        if is_correct is not None:
            if is_correct:
                print(f"✅ CORRECT")
            else:
                true_class_name = self.class_names[true_label]
                print(f"❌ INCORRECT (Expected: {true_class_name})")
        
        print("\nAll Class Scores:")
        print("-" * 40)
        
        # Sort by confidence
        sorted_indices = confidence_scores.argsort()[::-1]
        
        for idx in sorted_indices:
            class_name = self.class_names[idx]
            score = confidence_scores[idx]
            bar_length = int(score / 2)  # Scale to 50 chars max
            bar = '█' * bar_length + '░' * (50 - bar_length)
            
            marker = '👉' if idx == predicted_class_idx else '  '
            if true_label is not None and idx == true_label:
                marker = '✓ ' if idx == predicted_class_idx else '✗ '
            
            print(f"{marker} {class_name:20s} {bar} {score:6.2f}%")
        
        print("\n" + "="*70)
        
        # Create result dictionary
        result = {
            'video_name': video_name,
            'true_class': self.class_names[true_label] if true_label is not None else None,
            'predicted_class': predicted_class,
            'predicted_confidence': float(predicted_confidence),
            'is_correct': bool(is_correct) if is_correct is not None else None,  # Convert numpy.bool_ to Python bool
            'all_scores': {
                self.class_names[i]: float(confidence_scores[i])
                for i in range(len(self.class_names))
            },
            'timestamp': datetime.now().isoformat(),
            'tta_used': use_tta,
            'ensemble_size': len(self.models)
        }
        
        return result


def main():
    """Main execution - Interactive mode"""
    
    print("="*70)
    print("SINGLE VIDEO INFERENCE - INTERACTIVE MODE")
    print("Using pre-extracted features from features_enhanced folder or external video URL/local file")
    print("="*70)
    
    # Get current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Setup paths
    features_dir = script_dir.parent / 'video_classification_project' / 'features_enhanced'
    processed_test_dir = script_dir / 'data' / 'processed' / 'test'  # Use actual test structure
    models_dir = script_dir.parent / 'video_classification_project' / 'models_enhanced'
    
    # Check directories exist (features dir required for listing; external video branch can still work)
    if not features_dir.exists():
        print(f"\n✗ Error: features_enhanced directory not found at {features_dir}")
        sys.exit(1)
    
    if not processed_test_dir.exists():
        print(f"\n✗ Error: processed test directory not found at {processed_test_dir}")
        sys.exit(1)
    
    # Initialize feature loader
    feature_loader = SingleVideoFeatureLoader(
        features_dir=features_dir,
        processed_test_dir=processed_test_dir
    )
    
    # List available videos
    print("\n" + "="*70)
    print("STEP 1: SELECT VIDEO")
    print("="*70)
    
    all_videos, videos_by_category = feature_loader.list_available_videos()
    
    if not all_videos:
        print("\n✗ No videos found")
        sys.exit(1)
    
    video_input = input("\nEnter video number or filename (or 'test' to run accuracy test, or paste a URL/local path): ").strip()
    video_input = video_input.strip("'\"")
    
    if not video_input:
        print("No video specified")
        sys.exit(1)
    
    # Check if user wants to run accuracy test
    if video_input.lower() == 'test':
        print("\n" + "="*70)
        print("RUNNING ACCURACY TEST ON RANDOM SAMPLE")
        print("="*70)
        
        num_samples = input("\nHow many random videos to test? (default=20): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 20
        
        import random
        test_videos = random.sample(all_videos, min(num_samples, len(all_videos)))
        
        # Continue to model loading...
        run_batch_test = True
    else:
        run_batch_test = False
        
        # Check if input is a number
        video_name = None
        if video_input.isdigit():
            idx = int(video_input) - 1
            if 0 <= idx < len(all_videos):
                video_name = all_videos[idx]
                print(f"\n✓ Selected: {video_name}")
            else:
                print(f"\n✗ Invalid number. Must be between 1 and {len(all_videos)}")
                sys.exit(1)
        else:
            video_name = video_input
    
    # --- LOAD ONLY SELECTED CHECKPOINTS ---
    selected_names = [
        "best_ensemble_model_1.pt",
        "best_ensemble_model_2.pt",
        "best_ensemble_model_3.pt",
        "best_ensemble_model_4.pt",
    ]

    checkpoint_paths = []
    for name in selected_names:
        p = models_dir / name
        if p.exists():
            checkpoint_paths.append(str(p))
        else:
            print(f"⚠ Warning: checkpoint not found: {p.name}")

    if not checkpoint_paths:
        print(f"\n✗ Error: None of the selected checkpoints were found in {models_dir}")
        sys.exit(1)

    print(f"\nUsing selected {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"   - {Path(p).name}")

    classifier = SingleVideoClassifier(checkpoint_paths=checkpoint_paths, device='cuda')
    classifier.load_models()

    use_tta = False
    
    # Batch test mode
    if run_batch_test:
        print(f"\n{'='*70}")
        print(f"BATCH TESTING {len(test_videos)} VIDEOS")
        print(f"{'='*70}\n")
        
        use_tta = input("Use TTA for all tests? (y/n, default=n): ").strip().lower() == 'y'
        
        correct = 0
        total = 0
        results = []
        
        for i, video_name in enumerate(test_videos, 1):
            print(f"\n[{i}/{len(test_videos)}] Testing: {video_name}")
            
            try:
                # Load features
                features, label, category_name = feature_loader.load_video_features(video_name)
                
                if features is None:
                    print(f"  ✗ Could not load features")
                    continue
                
                # Convert label
                true_label_for_model = classifier.class_names.index(category_name)
                
                # Classify
                result = classifier.classify_video(
                    features=features,
                    true_label=true_label_for_model,
                    video_name=video_name,
                    use_tta=use_tta
                )
                
                results.append(result)
                
                if result['is_correct']:
                    correct += 1
                    print(f"  ✓ CORRECT: {result['predicted_class']} ({result['predicted_confidence']:.1f}%)")
                else:
                    print(f"  ✗ WRONG: Predicted {result['predicted_class']}, Expected {result['true_class']}")
                
                total += 1
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total tested: {total}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct}")
        print(f"Accuracy: {100*correct/total:.2f}%" if total > 0 else "N/A")
        print(f"{'='*70}\n")
        
        # Save results
        save = input("Save batch results to JSON? (y/n, default=n): ").strip().lower() == 'y'
        if save:
            output_file = f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total if total > 0 else 0
                    },
                    'results': results
                }, f, indent=2)
            print(f"✓ Results saved to: {output_file}")
        
        print("\n✅ Batch test complete!")
        sys.exit(0)
    
    # Single video or external URL/local file mode
    # Detect if input looks like a URL
    is_url = video_input.lower().startswith("http") or video_input.lower().startswith("www.")
    tmp_download_dir = None

    if is_url or (os.path.exists(video_input) and not video_input in all_videos):
        # treat as remote URL or direct local file path outside dataset
        try:
            print(f"\nProcessing external video: {video_input}")
            tmp_download_dir = tempfile.mkdtemp(prefix="external_video_")
            local_video = download_video_if_needed(video_input, tmp_dir=tmp_download_dir)
            print(f"Local video path: {local_video}")

            # Extract features using classifier.feature_dim
            model_feat_dim = classifier.feature_dim or 1280
            device_for_extraction = classifier.device
            print("Extracting frames and computing features (this may take time)...")
            features = extract_video_features(local_video, model_feature_dim=model_feat_dim, num_frames=73, device=device_for_extraction)
            print(f"Extracted features shape: {features.shape}")

            # classify (no true label)
            result = classifier.classify_video(
                features=features,
                true_label=None,
                video_name=local_video,
                use_tta=False
            )

            # Ask to save result
            save = input("\nSave result to JSON? (y/n, default=n): ").strip().lower() == 'y'
            if save:
                output_file = f"result_external_{Path(local_video).stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"✓ Result saved to: {output_file}")

            print("\n✅ Classification complete!")
        except Exception as e:
            print(f"\n✗ Error processing external video: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            if tmp_download_dir and os.path.exists(tmp_download_dir):
                shutil.rmtree(tmp_download_dir, ignore_errors=True)

        sys.exit(0)

    # Otherwise: original single-video-from-dataset flow (local indexed video)
    try:
        features, label, category_name = feature_loader.load_video_features(video_name)
        
        if features is None:
            print(f"\n✗ Error: Could not load features for '{video_name}'")
            sys.exit(1)
        
        # Convert category_name to model's label index
        true_category_name = category_name
            
    except Exception as e:
        print(f"\n✗ Error loading features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Convert label
    if true_category_name:
        try:
            true_label_for_model = classifier.class_names.index(true_category_name)
            print(f"\n✓ Label conversion:")
            print(f"  Category: {true_category_name}")
            print(f"  Model classes: {classifier.class_names}")
            print(f"  Model label index: {true_label_for_model}")
        except ValueError:
            print(f"\n⚠ Warning: Category '{true_category_name}' not found in model classes")
            true_label_for_model = None
    else:
        true_label_for_model = None
    
    # Classify
    try:
        result = classifier.classify_video(
            features=features,
            true_label=true_label_for_model,
            video_name=video_name,
            use_tta=use_tta
        )
        
        # Ask to save
        save = input("\nSave result to JSON? (y/n, default=n): ").strip().lower() == 'y'
        if save:
            output_file = f"result_{Path(video_name).stem}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Result saved to: {output_file}")
        
        print("\n✅ Classification complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
