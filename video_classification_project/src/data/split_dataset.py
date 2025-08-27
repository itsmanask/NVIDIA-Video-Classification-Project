import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

class VideoDatasetSplitter:
    def __init__(self, source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio  
        self.test_ratio = test_ratio
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"
    
    def split_videos_by_category(self):
        """
        Split videos according to the distribution table in your document
        """
        
        # Distribution from your document
        distribution = {
            'Animation': {
                'Cartoon': {'total': 562, 'train': 394, 'val': 112, 'test': 56},
                'Animation': {'total': 232, 'train': 163, 'val': 46, 'test': 23},
                'Pokemon': {'total': 47, 'train': 33, 'val': 9, 'test': 5}
            },
            'Gaming': {
                'Games': {'total': 498, 'train': 348, 'val': 100, 'test': 50},
                'Video_game': {'total': 341, 'train': 239, 'val': 68, 'test': 34}
            },
            'Natural_Content': {
                'Animal': {'total': 261, 'train': 183, 'val': 52, 'test': 26},
                'Pet': {'total': 220, 'train': 154, 'val': 44, 'test': 22}
            },
            'Flat_Content': {
                'Art': {'total': 357, 'train': 250, 'val': 71, 'test': 36},
                'Drawing': {'total': 220, 'train': 154, 'val': 44, 'test': 22}
            }
        }
        
        for main_category, subcategories in distribution.items():
            print(f"\nProcessing {main_category}...")
            
            for subcategory, splits in subcategories.items():
                source_path = self.source_dir / main_category / subcategory
                
                if not source_path.exists():
                    print(f"Warning: {source_path} not found, skipping...")
                    continue
                
                # Get all video files
                video_files = list(source_path.glob("*.mp4")) + \
                             list(source_path.glob("*.avi")) + \
                             list(source_path.glob("*.mov")) + \
                             list(source_path.glob("*.mkv"))
                
                if len(video_files) == 0:
                    print(f"No video files found in {source_path}")
                    continue
                
                # Shuffle for random split
                random.shuffle(video_files)
                
                # Calculate actual splits based on available files
                total_files = len(video_files)
                train_count = min(splits['train'], int(total_files * self.train_ratio))
                val_count = min(splits['val'], int(total_files * self.val_ratio))
                test_count = min(splits['test'], total_files - train_count - val_count)
                
                print(f"  {subcategory}: {total_files} files -> Train:{train_count}, Val:{val_count}, Test:{test_count}")
                
                # Split files
                train_files = video_files[:train_count]
                val_files = video_files[train_count:train_count + val_count]
                test_files = video_files[train_count + val_count:train_count + val_count + test_count]
                
                # Copy files to respective folders
                self._copy_files(train_files, 'train', main_category, subcategory)
                self._copy_files(val_files, 'val', main_category, subcategory)
                self._copy_files(test_files, 'test', main_category, subcategory)
    
    def _copy_files(self, files, split, main_category, subcategory):
        """Copy files to the target split directory"""
        target_path = self.target_dir / 'raw' / split / main_category / subcategory
        target_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            dest_path = target_path / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
    
    def verify_split(self):
        """Verify the split was successful"""
        print("\n=== SPLIT VERIFICATION ===")
        
        for split in ['train', 'val', 'test']:
            split_path = self.target_dir / 'raw' / split
            if split_path.exists():
                total_files = len(list(split_path.rglob("*.mp4"))) + \
                             len(list(split_path.rglob("*.avi"))) + \
                             len(list(split_path.rglob("*.mov"))) + \
                             len(list(split_path.rglob("*.mkv")))
                print(f"{split.upper()}: {total_files} files")
                
                # Count by category
                for main_cat in ['Animation', 'Gaming', 'Natural_Content', 'Flat_Content']:
                    cat_path = split_path / main_cat
                    if cat_path.exists():
                        cat_files = len(list(cat_path.rglob("*.mp4"))) + \
                                   len(list(cat_path.rglob("*.avi"))) + \
                                   len(list(cat_path.rglob("*.mov"))) + \
                                   len(list(cat_path.rglob("*.mkv")))
                        print(f"  {main_cat}: {cat_files}")

# Usage example
def main():
    # MODIFY THESE PATHS TO MATCH YOUR SETUP
    source_directory = "D:\\PR1\\Dataset"  # Where your 4000 videos are currently stored
    target_directory = "video_classification_project/data"
    
    splitter = VideoDatasetSplitter(
        source_dir=source_directory,
        target_dir=target_directory,
        train_ratio=0.7,
        val_ratio=0.2, 
        test_ratio=0.1
    )
    
    print("Starting video dataset split...")
    splitter.split_videos_by_category()
    splitter.verify_split()
    print("\nDataset split completed!")

if __name__ == "__main__":
    main()