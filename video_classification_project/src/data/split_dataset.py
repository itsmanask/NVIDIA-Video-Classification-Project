import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

class VideoDatasetSplitter:
    def __init__(self, source_base_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.source_base_dir = Path(source_base_dir)  # D:\PR1\Dataset
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio  
        self.test_ratio = test_ratio
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"
    
    def split_videos_by_category(self):
        """
        Split videos from your subcategorized directory structure:
        Dataset/Animation/videos/Cartoon/
        Dataset/Animation/videos/Pokemon/
        Dataset/Gaming/videos/Games/
        etc.
        """
        
        # Define all subcategories based on your actual structure
        category_structure = {
            'Animation': ['Cartoon', 'Animation', 'Lego minifigure', 'Naruto', 'The Walt Disney Company', 
                         'Dragon Ball', 'Sonic the Hedgehog', 'One Piece', 'Walt Disney World', 
                         'Bleach', 'Mickey Mouse'],
            
            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty', 'Grand Theft Auto V', 
                      'World of Warcraft', 'Call of Duty: Black Ops II', 'League of Legends', 
                      'Battlefield', 'Grand Theft Auto: San Andreas', 'RuneScape', 
                      'Call of Duty: Modern Warfare 3', 'Call of Duty: Black Ops', 'FIFA 15', 
                      'Counter-Strike', 'Need for Speed'],
            
            'Natural Content': ['Animal', 'Pet', 'Fishing', 'Fish', 'Outdoor recreation', 'Dog', 
                               'Horse', 'Bird', 'Plant', 'Cat', 'Farm', 'Garden', 'Nature', 
                               'Tree', 'Wildlife', 'Chicken', 'Lion', 'Deer', 'Bear', 'Elephant'],
            
            'Flat Content': ['Website', 'Chart', 'Map', 'Logo', 'Text', 'Typography', 
                        'Screencast', 'Illustration', 'Poster']
        }
        
        # Map original folder names to standardized names
        name_mapping = {
            'Natural Content': 'Natural_Content',
            'Flat Content': 'Flat_Content',
            'Animation': 'Animation',
            'Gaming': 'Gaming'
        }
        
        for original_category, subcategories in category_structure.items():
            standard_category = name_mapping[original_category]
            print(f"\nProcessing {original_category}...")
            
            # Collect all videos from all subcategories of this main category
            all_category_videos = []
            
            for subcategory in subcategories:
                subcategory_path = self.source_base_dir / original_category / 'videos' / subcategory
                
                if not subcategory_path.exists():
                    print(f"  Warning: {subcategory_path} not found, skipping...")
                    continue
                
                # Get all video files from this subcategory - fix double counting
                video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
                subcategory_videos = []
                
                for ext in video_extensions:
                    files = list(subcategory_path.glob(ext))
                    subcategory_videos.extend(files)
                
                # Remove any potential duplicates (though there shouldn't be any)
                subcategory_videos = list(set(subcategory_videos))
                
                if subcategory_videos:
                    print(f"  Found {len(subcategory_videos)} videos in {subcategory}")
                    all_category_videos.extend(subcategory_videos)
            
            if not all_category_videos:
                print(f"  No videos found in {original_category}")
                continue
            
            # Shuffle all videos from this category for random split
            random.shuffle(all_category_videos)
            
            total_files = len(all_category_videos)
            print(f"  Total {original_category} videos: {total_files}")
            
            # Calculate splits based on ratios
            train_count = int(total_files * self.train_ratio)
            val_count = int(total_files * self.val_ratio)
            test_count = total_files - train_count - val_count
            
            print(f"  Splitting: Train:{train_count}, Val:{val_count}, Test:{test_count}")
            
            # Split files
            train_files = all_category_videos[:train_count]
            val_files = all_category_videos[train_count:train_count + val_count]
            test_files = all_category_videos[train_count + val_count:]
            
            # Copy files to respective folders (using standard category names)
            self._copy_files(train_files, 'train', standard_category)
            self._copy_files(val_files, 'val', standard_category)
            self._copy_files(test_files, 'test', standard_category)
    
    def _copy_files(self, files, split, main_category):
        """Copy files to the target split directory"""
        target_path = self.target_dir / 'raw' / split / main_category
        target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"    Copying {len(files)} files to {target_path}")
        
        for i, file_path in enumerate(files):
            dest_path = target_path / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 50 == 0:  # Progress indicator
                    print(f"      Copied {i + 1}/{len(files)} files")
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
    
    def verify_split(self):
        """Verify the split was successful"""
        print("\n=== SPLIT VERIFICATION ===")
        
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
        
        for split in ['train', 'val', 'test']:
            split_path = self.target_dir / 'raw' / split
            if split_path.exists():
                # Count total files
                total_files = 0
                for ext in video_extensions:
                    total_files += len(list(split_path.rglob(ext)))
                
                print(f"{split.upper()}: {total_files} files")
                
                # Count by category
                for main_cat in ['Animation', 'Gaming', 'Natural_Content', 'Flat_Content']:
                    cat_path = split_path / main_cat
                    if cat_path.exists():
                        cat_files = 0
                        for ext in video_extensions:
                            cat_files += len(list(cat_path.glob(ext)))
                        print(f"  {main_cat}: {cat_files}")
            else:
                print(f"{split.upper()}: Directory not found")

# Usage example
def main():
    # Use relative paths from current working directory (PR1)
    source_directory = "Dataset"  # Assumes running from PR1 directory
    target_directory = "video_classification_project/data"  # Will be created in PR1
    
    print("Starting video dataset split...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Source: {Path(source_directory).absolute()}")
    print(f"Target: {Path(target_directory).absolute()}")
    print()
    
    # Check if source directory exists
    if not Path(source_directory).exists():
        print(f"ERROR: Source directory '{source_directory}' not found!")
        print("Please ensure you're running this script from the PR1 directory")
        print("and that the 'Dataset' folder exists.")
        return
    
    splitter = VideoDatasetSplitter(
        source_base_dir=source_directory,
        target_dir=target_directory,
        train_ratio=0.7,
        val_ratio=0.2, 
        test_ratio=0.1
    )
    
    splitter.split_videos_by_category()
    splitter.verify_split()
    print("\nDataset split completed!")

if __name__ == "__main__":
    main()