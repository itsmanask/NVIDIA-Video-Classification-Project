#VideoDatasetSplitter class that allows splitting only specified categories (e.g., only "Animation") without re-splitting already processed categories

import os
import shutil
import random
from pathlib import Path


class VideoDatasetSplitter:
    def __init__(self, source_base_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.source_base_dir = Path(source_base_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"

    def split_videos_by_category(self, categories_to_split=None):
        """
        Split videos from subcategorized directory structure.
        Optionally, only split the categories listed in categories_to_split.
        If categories_to_split is None, splits all categories.
        """

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

        name_mapping = {
            'Natural Content': 'Natural_Content',
            'Flat Content': 'Flat_Content',
            'Animation': 'Animation',
            'Gaming': 'Gaming'
        }

        # Determine which categories to process
        categories = category_structure.keys()
        if categories_to_split:
            categories = [cat for cat in categories if cat in categories_to_split]

        for original_category in categories:
            subcategories = category_structure[original_category]
            standard_category = name_mapping[original_category]
            print(f"\nProcessing {original_category}...")

            all_category_videos = []

            for subcategory in subcategories:
                subcategory_path = self.source_base_dir / original_category / 'videos' / subcategory

                if not subcategory_path.exists():
                    print(f"   Warning: {subcategory_path} not found, skipping...")
                    continue

                video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
                subcategory_videos = []

                for ext in video_extensions:
                    files = list(subcategory_path.glob(ext))
                    subcategory_videos.extend(files)

                # Remove duplicates if any
                subcategory_videos = list(set(subcategory_videos))

                if subcategory_videos:
                    print(f"   Found {len(subcategory_videos)} videos in {subcategory}")
                    all_category_videos.extend(subcategory_videos)

            if not all_category_videos:
                print(f"   No videos found in {original_category}")
                continue

            # Shuffle for randomness
            random.shuffle(all_category_videos)

            total_files = len(all_category_videos)
            print(f"   Total {original_category} videos: {total_files}")

            train_count = int(total_files * self.train_ratio)
            val_count = int(total_files * self.val_ratio)
            test_count = total_files - train_count - val_count

            print(f"   Splitting: Train:{train_count}, Val:{val_count}, Test:{test_count}")

            train_files = all_category_videos[:train_count]
            val_files = all_category_videos[train_count:train_count + val_count]
            test_files = all_category_videos[train_count + val_count:]

            self._copy_files(train_files, 'train', standard_category)
            self._copy_files(val_files, 'val', standard_category)
            self._copy_files(test_files, 'test', standard_category)

    def _copy_files(self, files, split, main_category):
        """Copy files to the target directory"""
        target_path = self.target_dir / 'raw' / split / main_category
        target_path.mkdir(parents=True, exist_ok=True)

        print(f"     Copying {len(files)} files to {target_path}")

        for i, file_path in enumerate(files):
            dest_path = target_path / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 50 == 0:
                    print(f"       Copied {i + 1}/{len(files)} files")
            except Exception as e:
                print(f"Error copying {file_path}: {e}")

    def verify_split(self):
        """Print counts of files in each split and category for verification"""
        print("\n=== SPLIT VERIFICATION ===")
        
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]

        for split in ['train', 'val', 'test']:
            split_path = self.target_dir / 'raw' / split
            if split_path.exists():
                total_files = 0
                for ext in video_extensions:
                    total_files += len(list(split_path.rglob(ext)))
                print(f"{split.upper()}: {total_files} files")

                for main_cat in ['Animation', 'Gaming', 'Natural_Content', 'Flat_Content']:
                    cat_path = split_path / main_cat
                    if cat_path.exists():
                        cat_files = 0
                        for ext in video_extensions:
                            cat_files += len(list(cat_path.glob(ext)))
                        print(f"   {main_cat}: {cat_files}")
            else:
                print(f"{split.upper()}: Directory not found")


# Usage example: splitting only Animation category without re-splitting others
def main():
    source_directory = "Dataset"  # relative path to your dataset dir
    target_directory = "video_classification_project/data"  # output target dir

    print("Starting video dataset split...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Source: {Path(source_directory).absolute()}")
    print(f"Target: {Path(target_directory).absolute()}")
    print()

    if not Path(source_directory).exists():
        print(f"ERROR: Source directory '{source_directory}' not found!")
        print("Please ensure you're running this script from the correct directory")
        print("and that the 'Dataset' folder exists.")
        return

    splitter = VideoDatasetSplitter(
        source_base_dir=source_directory,
        target_dir=target_directory,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    # Pass list with only "Animation" to split only that category
    splitter.split_videos_by_category(categories_to_split=['Animation'])
    splitter.verify_split()
    print("\nDataset split completed!")


if __name__ == "__main__":
    main()
