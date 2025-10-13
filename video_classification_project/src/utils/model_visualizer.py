import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import cv2

# Import your model
from model_train import EnhancedCNNLSTM, MemoryEfficientVideoDataset


class DataModelVisualizer:
    """Visualize preprocessed data and model predictions"""
    
    def __init__(self, data_dir, checkpoint_path=None, output_dir='visualizations'):
        self.data_dir = Path(data_dir)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.category_mapping = None
        
        # ImageNet mean and std for denormalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"Visualizer initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def denormalize_frame(self, tensor):
        """Convert normalized tensor back to displayable image"""
        # tensor shape: [C, H, W]
        img = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        img = img * self.std + self.mean
        img = np.clip(img, 0, 1)
        return img
    
    def load_model(self):
        """Load trained model"""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            print("No checkpoint provided, skipping model loading")
            return False
        
        print(f"\nLoading model from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get num_classes from checkpoint or use default
        num_classes = 4  # Default
        
        self.model = EnhancedCNNLSTM(
            num_classes=num_classes,
            hidden_dim=512,
            num_layers=3,
            dropout=0.25,
            backbone='resnet50',
            bidirectional=True,
            attention=True
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"  Best Val Acc: {checkpoint['metrics'].get('best_val_acc', 'N/A')}")
        
        return True
    
    def visualize_single_video(self, video_tensor, label, filename, save_name):
        """Visualize all frames of a single video"""
        num_frames = video_tensor.shape[0]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 10))
        
        # Main subplot: 8x4 grid for all 32 frames
        for i in range(min(32, num_frames)):
            ax = plt.subplot(4, 8, i + 1)
            img = self.denormalize_frame(video_tensor[i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Frame {i+1}', fontsize=8)
        
        plt.suptitle(f'Video: {filename}\nLabel: {label}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}")
    
    def visualize_category_samples(self, split='train', samples_per_category=3):
        """Visualize sample videos from each category"""
        print(f"\n{'='*60}")
        print(f"VISUALIZING {split.upper()} SPLIT SAMPLES")
        print(f"{'='*60}")
        
        split_dir = self.data_dir / split
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            return
        
        # Scan all categories
        categories = sorted([d for d in split_dir.glob("*") if d.is_dir()])
        
        for category_dir in categories:
            category_name = category_dir.name
            print(f"\nCategory: {category_name}")
            
            # Find subcategories
            subcategories = sorted([d for d in category_dir.glob("*") if d.is_dir()])
            
            sample_count = 0
            for subcat_dir in subcategories:
                if sample_count >= samples_per_category:
                    break
                
                data_file = subcat_dir / 'processed_data.pt'
                if not data_file.exists():
                    continue
                
                # Load data
                try:
                    data = torch.load(data_file, map_location='cpu')
                    videos = data['videos']
                    labels = data['labels']
                    filenames = data.get('filenames', [f"video_{i}" for i in range(len(videos))])
                    
                    # Get category mapping
                    if self.category_mapping is None:
                        self.category_mapping = data.get('category_mapping', {})
                    
                    # Sample a few videos
                    num_samples = min(samples_per_category - sample_count, len(videos))
                    
                    for i in range(num_samples):
                        video = videos[i]
                        label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                        filename = filenames[i] if i < len(filenames) else f"video_{i}"
                        
                        save_name = f"{split}_{category_name}_{subcat_dir.name}_sample{i+1}.png"
                        self.visualize_single_video(video, label, filename, save_name)
                        
                        sample_count += 1
                        
                        if sample_count >= samples_per_category:
                            break
                
                except Exception as e:
                    print(f"Error loading {data_file}: {e}")
                    continue
    
    def create_frame_montage(self, video_tensor, cols=8):
        """Create a montage of all frames in a video"""
        num_frames = video_tensor.shape[0]
        rows = (num_frames + cols - 1) // cols
        
        # Get frame dimensions
        _, h, w = video_tensor[0].shape
        
        # Create montage
        montage = np.zeros((rows * h, cols * w, 3))
        
        for i in range(num_frames):
            row = i // cols
            col = i % cols
            img = self.denormalize_frame(video_tensor[i])
            montage[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return montage
    
    def visualize_predictions(self, split='val', num_samples=10):
        """Visualize model predictions on samples"""
        if self.model is None:
            print("Model not loaded. Please provide checkpoint_path.")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING PREDICTIONS FOR {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = MemoryEfficientVideoDataset(
            self.data_dir,
            split=split,
            max_memory_gb=2.0,
            cache_size=10
        )
        
        if len(dataset) == 0:
            print(f"No data found in {split} split")
            return
        
        # Get category names
        category_names = [None] * len(dataset.category_mapping)
        for name, idx in dataset.category_mapping.items():
            category_names[idx] = name
        
        print(f"Found {len(dataset)} samples")
        print(f"Categories: {category_names}")
        
        # Sample random videos
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Generating predictions"):
                video, true_label = dataset[idx]
                
                # Add batch dimension
                video_batch = video.unsqueeze(0).to(self.device)
                
                # Predict
                output = self.model(video_batch)
                probs = torch.softmax(output, dim=1)[0]
                pred_label = output.argmax(1).item()
                
                # Visualize
                self.visualize_prediction_result(
                    video, 
                    true_label.item(), 
                    pred_label, 
                    probs.cpu().numpy(),
                    category_names,
                    f"{split}_prediction_sample_{idx}.png"
                )
    
    def visualize_prediction_result(self, video, true_label, pred_label, probs, 
                                   category_names, save_name):
        """Visualize prediction with frame montage and probabilities"""
        fig = plt.figure(figsize=(16, 10))
        
        # Top: Frame montage (first 16 frames)
        ax1 = plt.subplot(2, 1, 1)
        montage = self.create_frame_montage(video[:16], cols=8)
        ax1.imshow(montage)
        ax1.axis('off')
        ax1.set_title('Video Frames (First 16)', fontsize=14, fontweight='bold')
        
        # Bottom: Prediction results
        ax2 = plt.subplot(2, 1, 2)
        
        # Bar chart of probabilities
        y_pos = np.arange(len(category_names))
        colors = ['green' if i == pred_label else 'red' if i == true_label else 'gray' 
                  for i in range(len(category_names))]
        
        bars = ax2.barh(y_pos, probs, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(category_names)
        ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob*100:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Title with prediction result
        correct = pred_label == true_label
        result_text = "✓ CORRECT" if correct else "✗ INCORRECT"
        result_color = 'green' if correct else 'red'
        
        title = f"Prediction: {category_names[pred_label]} | True: {category_names[true_label]} | {result_text}"
        ax2.set_title(title, fontsize=14, fontweight='bold', 
                     color=result_color, pad=20)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Predicted'),
            Patch(facecolor='red', alpha=0.7, label='True Label'),
            Patch(facecolor='gray', alpha=0.7, label='Other')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_frame_progression(self, split='train', category=None, num_videos=5):
        """Analyze how frames progress through a video"""
        print(f"\n{'='*60}")
        print(f"ANALYZING FRAME PROGRESSION")
        print(f"{'='*60}")
        
        split_dir = self.data_dir / split
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            return
        
        # Find category
        if category:
            category_dirs = [split_dir / category]
        else:
            category_dirs = sorted([d for d in split_dir.glob("*") if d.is_dir()])[:1]
        
        for category_dir in category_dirs:
            category_name = category_dir.name
            print(f"\nAnalyzing category: {category_name}")
            
            # Find first subcategory with data
            subcategories = sorted([d for d in category_dir.glob("*") if d.is_dir()])
            
            for subcat_dir in subcategories:
                data_file = subcat_dir / 'processed_data.pt'
                if not data_file.exists():
                    continue
                
                try:
                    data = torch.load(data_file, map_location='cpu')
                    videos = data['videos']
                    
                    # Analyze first few videos
                    for vid_idx in range(min(num_videos, len(videos))):
                        video = videos[vid_idx]
                        
                        # Create progression visualization
                        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
                        fig.suptitle(f'Frame Progression: {category_name} - Video {vid_idx+1}', 
                                   fontsize=16, fontweight='bold')
                        
                        for frame_idx in range(32):
                            row = frame_idx // 8
                            col = frame_idx % 8
                            
                            img = self.denormalize_frame(video[frame_idx])
                            axes[row, col].imshow(img)
                            axes[row, col].axis('off')
                            axes[row, col].set_title(f'{frame_idx+1}', fontsize=8)
                        
                        plt.tight_layout()
                        save_path = self.output_dir / f"progression_{category_name}_video{vid_idx+1}.png"
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Saved progression for video {vid_idx+1}")
                    
                    break  # Only process first subcategory
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            break  # Only process first category
    
    def create_dataset_overview(self):
        """Create comprehensive dataset overview"""
        print(f"\n{'='*60}")
        print(f"CREATING DATASET OVERVIEW")
        print(f"{'='*60}")
        
        stats = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        # Collect statistics
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            
            print(f"\nScanning {split} split...")
            
            categories = sorted([d for d in split_dir.glob("*") if d.is_dir()])
            
            for category_dir in categories:
                category_name = category_dir.name
                
                if category_name not in stats[split]:
                    stats[split][category_name] = 0
                
                subcategories = sorted([d for d in category_dir.glob("*") if d.is_dir()])
                
                for subcat_dir in subcategories:
                    data_file = subcat_dir / 'processed_data.pt'
                    if data_file.exists():
                        try:
                            data = torch.load(data_file, map_location='cpu')
                            num_videos = len(data['videos'])
                            stats[split][category_name] += num_videos
                        except:
                            continue
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Category distribution per split
        ax = axes[0, 0]
        all_categories = set()
        for split_stats in stats.values():
            all_categories.update(split_stats.keys())
        all_categories = sorted(list(all_categories))
        
        x = np.arange(len(all_categories))
        width = 0.25
        
        for i, split in enumerate(['train', 'val', 'test']):
            counts = [stats[split].get(cat, 0) for cat in all_categories]
            ax.bar(x + i*width, counts, width, label=split.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Category', fontweight='bold')
        ax.set_ylabel('Number of Videos', fontweight='bold')
        ax.set_title('Videos per Category by Split', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Total videos per split
        ax = axes[0, 1]
        split_totals = {split: sum(stats[split].values()) for split in ['train', 'val', 'test']}
        colors = ['#3b82f6', '#8b5cf6', '#ec4899']
        wedges, texts, autotexts = ax.pie(
            split_totals.values(), 
            labels=[f'{k.capitalize()}\n{v}' for k, v in split_totals.items()],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('Dataset Split Distribution', fontweight='bold', fontsize=14)
        
        # 3. Category proportions
        ax = axes[1, 0]
        category_totals = {}
        for split_stats in stats.values():
            for cat, count in split_stats.items():
                category_totals[cat] = category_totals.get(cat, 0) + count
        
        if category_totals:
            colors_cat = ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24']
            ax.barh(list(category_totals.keys()), list(category_totals.values()), 
                   color=colors_cat, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Total Videos', fontweight='bold')
            ax.set_title('Total Videos per Category', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
        
        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        total_videos = sum(split_totals.values())
        total_frames = total_videos * 32
        
        stats_text = [
            ["Metric", "Value"],
            ["─" * 30, "─" * 15],
            ["Total Videos", f"{total_videos:,}"],
            ["Total Frames", f"{total_frames:,}"],
            ["Categories", f"{len(all_categories)}"],
            ["Training Videos", f"{split_totals.get('train', 0):,}"],
            ["Validation Videos", f"{split_totals.get('val', 0):,}"],
            ["Test Videos", f"{split_totals.get('test', 0):,}"],
            ["Frames per Video", "32"],
            ["Frame Size", "224 × 224"],
            ["Normalization", "ImageNet"]
        ]
        
        table = ax.table(cellText=stats_text, loc='center', cellLoc='left',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#374151')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style alternating rows
        for i in range(2, len(stats_text)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f3f4f6')
        
        ax.set_title('Dataset Statistics', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / 'dataset_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Dataset overview saved: {save_path}")
        
        # Print text summary
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        for row in stats_text[2:]:  # Skip header
            print(f"  {row[0]:<25} {row[1]:>15}")
    
    def compare_augmentation_effects(self):
        """Compare augmented vs non-augmented samples"""
        print(f"\n{'='*60}")
        print(f"COMPARING AUGMENTATION EFFECTS")
        print(f"{'='*60}")
        
        # Load one video from train (augmented) and val (not augmented)
        for split in ['train', 'val']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            
            categories = sorted([d for d in split_dir.glob("*") if d.is_dir()])
            
            for category_dir in categories[:1]:  # First category only
                subcategories = sorted([d for d in category_dir.glob("*") if d.is_dir()])
                
                for subcat_dir in subcategories[:1]:  # First subcategory only
                    data_file = subcat_dir / 'processed_data.pt'
                    if not data_file.exists():
                        continue
                    
                    try:
                        data = torch.load(data_file, map_location='cpu')
                        video = data['videos'][0]  # First video
                        
                        # Show first 8 frames
                        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                        fig.suptitle(f'{split.upper()} Split Sample - {"Augmented" if split == "train" else "Original"}', 
                                   fontsize=16, fontweight='bold')
                        
                        for i in range(8):
                            row = i // 4
                            col = i % 4
                            
                            img = self.denormalize_frame(video[i])
                            axes[row, col].imshow(img)
                            axes[row, col].axis('off')
                            axes[row, col].set_title(f'Frame {i+1}', fontsize=10)
                        
                        plt.tight_layout()
                        save_path = self.output_dir / f'augmentation_comparison_{split}.png'
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Saved {split} sample")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                    
                    break
                break


def main():
    """Interactive visualization menu"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize preprocessed data and model predictions')
    parser.add_argument('--data_dir', type=str, 
                       default='video_classification_project/data/processed',
                       help='Path to preprocessed data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VIDEO CLASSIFICATION VISUALIZER")
    print("="*60)
    
    # Initialize visualizer
    visualizer = DataModelVisualizer(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Load model if checkpoint provided
    if args.checkpoint:
        visualizer.load_model()
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("VISUALIZATION OPTIONS")
        print("="*60)
        print("1. Dataset Overview (statistics and distribution)")
        print("2. Visualize Sample Videos (3 per category)")
        print("3. Analyze Frame Progression")
        print("4. Compare Augmentation Effects")
        print("5. Model Predictions (requires checkpoint)")
        print("6. Run All Visualizations")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                print("\nExiting visualizer. Goodbye!")
                break
            
            elif choice == '1':
                visualizer.create_dataset_overview()
            
            elif choice == '2':
                split = input("Select split (train/val/test) [default: train]: ").strip() or 'train'
                samples = int(input("Samples per category [default: 3]: ").strip() or '3')
                visualizer.visualize_category_samples(split=split, samples_per_category=samples)
            
            elif choice == '3':
                split = input("Select split (train/val/test) [default: train]: ").strip() or 'train'
                num_videos = int(input("Number of videos to analyze [default: 5]: ").strip() or '5')
                visualizer.analyze_frame_progression(split=split, num_videos=num_videos)
            
            elif choice == '4':
                visualizer.compare_augmentation_effects()
            
            elif choice == '5':
                if visualizer.model is None:
                    print("\nNo model loaded! Please provide checkpoint path.")
                else:
                    split = input("Select split (val/test) [default: val]: ").strip() or 'val'
                    num_samples = int(input("Number of samples [default: 10]: ").strip() or '10')
                    visualizer.visualize_predictions(split=split, num_samples=num_samples)
            
            elif choice == '6':
                print("\nRunning all visualizations...")
                visualizer.create_dataset_overview()
                visualizer.visualize_category_samples(split='train', samples_per_category=2)
                visualizer.analyze_frame_progression(split='train', num_videos=3)
                visualizer.compare_augmentation_effects()
                if visualizer.model is not None:
                    visualizer.visualize_predictions(split='val', num_samples=10)
                print("\n✓ All visualizations complete!")
            
            else:
                print("Invalid option. Please select 0-6.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()