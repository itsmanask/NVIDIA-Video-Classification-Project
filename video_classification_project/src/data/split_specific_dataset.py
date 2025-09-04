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

        self.category_structure = {
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

        self.name_mapping = {
            'Natural Content': 'Natural_Content',
            'Flat Content': 'Flat_Content',
            'Animation': 'Animation',
            'Gaming': 'Gaming'
        }

        self.video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]

    def split_videos_by_category(self, categories_to_split=None):
        """
        Split videos and preserve subcategory structure inside train/val/test.
        """

        # Decide which categories to split
        categories = list(self.category_structure.keys())
        if categories_to_split:
            categories = [cat for cat in categories if cat in categories_to_split]

        for original_category in categories:
            standard_category = self.name_mapping[original_category]
            subcategories = self.category_structure[original_category]

            print(f"\nProcessing category: {original_category} ...")

            for subcategory in subcategories:
                subcategory_path = self.source_base_dir / original_category / "videos" / subcategory

                if not subcategory_path.exists():
                    print(f"   WARNING: {subcategory_path} not found, skipping...")
                    continue

                # Collect videos from subcategory
                videos = []
                for ext in self.video_extensions:
                    videos.extend(list(subcategory_path.glob(ext)))
                videos = list(set(videos))

                if not videos:
                    print(f"   No videos found in {subcategory}, skipping...")
                    continue

                print(f"   Found {len(videos)} videos in {subcategory}")

                # Shuffle for randomness
                random.shuffle(videos)

                # Split train/val/test counts
                total = len(videos)
                train_count = int(total * self.train_ratio)
                val_count = int(total * self.val_ratio)
                test_count = total - train_count - val_count

                train_files = videos[:train_count]
                val_files = videos[train_count:train_count + val_count]
                test_files = videos[train_count + val_count:]

                # Copy to subcategory-preserving structure
                self._copy_files(train_files, 'train', standard_category, subcategory)
                self._copy_files(val_files, 'val', standard_category, subcategory)
                self._copy_files(test_files, 'test', standard_category, subcategory)

    def _copy_files(self, files, split, main_category, subcategory):
        """Copy to target/raw/{split}/{main_category}/{subcategory}"""
        target_path = self.target_dir / "raw" / split / main_category / subcategory
        target_path.mkdir(parents=True, exist_ok=True)

        print(f"     Copying {len(files)} files → {target_path}")

        for i, file_path in enumerate(files):
            dest_path = target_path / file_path.name
            if dest_path.exists():  # Avoid re-splitting already processed files
                continue
            try:
                shutil.copy2(file_path, dest_path)
                if (i + 1) % 50 == 0:
                    print(f"       Copied {i+1}/{len(files)} files")
            except Exception as e:
                print(f"Error copying {file_path}: {e}")

    def verify_split(self):
        """Verify file counts after splitting"""
        print("\n=== SPLIT VERIFICATION ===")

        for split in ['train', 'val', 'test']:
            split_path = self.target_dir / 'raw' / split
            print(f"\n{split.upper()}:")

            if not split_path.exists():
                print("   Directory not found")
                continue

            for main_cat in ['Animation', 'Gaming', 'Natural_Content', 'Flat_Content']:
                cat_path = split_path / main_cat
                if not cat_path.exists():
                    continue
                main_count = sum(len(list(cat_path.rglob(ext))) for ext in self.video_extensions)
                print(f"   {main_cat}: {main_count} files")

                for subcat in cat_path.iterdir():
                    if subcat.is_dir():
                        sub_count = sum(len(list(subcat.glob(ext))) for ext in self.video_extensions)
                        print(f"      {subcat.name}: {sub_count}")


# ------------------------- Usage Example -------------------------
def main():
    source_directory = r"C:\Users\rajla\NVIDIA-Video-Classification-Project\Dataset"
    target_directory = r"C:\Users\rajla\NVIDIA-Video-Classification-Project\video_classification_project\data"

    print("Starting dataset split...")
    splitter = VideoDatasetSplitter(source_directory, target_directory, 0.7, 0.2, 0.1)

    # (1) To split ALL 4 categories
    splitter.split_videos_by_category()

    # (2) To split only Animation (without touching others)
    # splitter.split_videos_by_category(categories_to_split=['Animation'])

    splitter.verify_split()
    print("\nDataset split completed!")


if __name__ == "__main__":
    main()
