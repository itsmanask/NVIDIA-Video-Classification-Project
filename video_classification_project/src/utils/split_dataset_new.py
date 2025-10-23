import os
import shutil
import random
from pathlib import Path
import json


class VideoDatasetSplitter:
    def __init__(self, source_base_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.source_base_dir = Path(source_base_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Set random seed for reproducible splits
        random.seed(self.seed)
        print(f"Random seed set to: {self.seed}")

        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"

        self.category_structure = {
            'Animation': ['Cartoon', 'Animation', 'Lego minifigure', 'Naruto', 'The Walt Disney Company',
                          'Dragon Ball', 'Sonic the Hedgehog', 'One Piece', 'Bleach'],

            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty','Grand Theft Auto', 'Grand Theft Auto V',
                       'World of Warcraft', 'League of Legends', 'Battlefield', 'RuneScape',
                       'Action-adventure game', 'FIFA 15'],

            'Natural Content': ['Animal', 'Pet', 'Fishing', 'Fish', 'Outdoor recreation', 'Dog',
                                'Horse', 'Bird', 'Plant', 'Cat', 'Farm', 'Garden', 'Nature',
                                'Tree', 'Wildlife', 'Chicken'],

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
        
        # Track processed files to prevent duplicates
        self.processed_files = self.load_processed_files()
        self.split_manifest_path = self.target_dir / "raw" / "split_manifest.json"

    def load_processed_files(self):
        """Load the list of already processed files from manifest"""
        manifest_path = self.target_dir / "raw" / "split_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                    print(f"\nLoaded manifest with {len(data.get('files', {}))} already processed files")
                    return data.get('files', {})
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
                return {}
        return {}

    def save_processed_files(self):
        """Save the manifest of processed files"""
        manifest_path = self.target_dir / "raw" / "split_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest_data = {
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'files': self.processed_files
        }
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            print(f"\nManifest saved: {len(self.processed_files)} files tracked")
        except Exception as e:
            print(f"Warning: Could not save manifest: {e}")

    def is_file_already_processed(self, file_path):
        """Check if a file has already been processed"""
        file_key = file_path.name  # Use filename as key
        return file_key in self.processed_files

    def mark_file_as_processed(self, file_path, split, category, subcategory):
        """Mark a file as processed in the specified split"""
        file_key = file_path.name
        self.processed_files[file_key] = {
            'split': split,
            'category': category,
            'subcategory': subcategory,
            'source_path': str(file_path)
        }

    def check_existing_splits(self):
        """Check if there are already split files in the target directory"""
        split_dirs = ['train', 'val', 'test']
        existing_files = 0
        
        for split in split_dirs:
            split_path = self.target_dir / 'raw' / split
            if split_path.exists():
                for ext in self.video_extensions:
                    existing_files += len(list(split_path.rglob(ext)))
        
        if existing_files > 0:
            print(f"\n{'='*70}")
            print(f"WARNING: Found {existing_files} existing files in target directory!")
            print(f"{'='*70}")
            print("Options:")
            print("  1. Continue (skip already processed files)")
            print("  2. Clean and restart (delete all existing splits)")
            print("  3. Cancel")
            
            while True:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice == '1':
                    print("\nContinuing with existing files...")
                    return True
                elif choice == '2':
                    print("\nCleaning existing splits...")
                    self.clean_target_directory()
                    return True
                elif choice == '3':
                    print("\nOperation cancelled.")
                    return False
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
        
        return True

    def clean_target_directory(self):
        """Remove all existing split directories"""
        split_dirs = ['train', 'val', 'test']
        for split in split_dirs:
            split_path = self.target_dir / 'raw' / split
            if split_path.exists():
                print(f"  Removing {split_path}...")
                shutil.rmtree(split_path)
        
        # Clear manifest
        self.processed_files = {}
        manifest_path = self.target_dir / "raw" / "split_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        
        print("✓ Target directory cleaned")

    def get_user_category_choice(self):
        """Display menu and get user's category selection"""
        print("\n" + "="*50)
        print("VIDEO DATASET SPLITTER - CATEGORY SELECTION")
        print("="*50)
        
        categories = list(self.category_structure.keys())
        
        print("\nAvailable categories to split:")
        for i, category in enumerate(categories, 1):
            subcategory_count = len(self.category_structure[category])
            print(f"  {i}. {category} ({subcategory_count} subcategories)")
        
        print(f"  {len(categories) + 1}. All categories")
        print(f"  {len(categories) + 2}. Multiple categories (custom selection)")
        print("  0. Exit")
        
        while True:
            try:
                choice = input(f"\nEnter your choice (0-{len(categories) + 2}): ").strip()
                
                if choice == '0':
                    print("Exiting...")
                    return None
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(categories):
                    selected_category = categories[choice_num - 1]
                    print(f"\nSelected: {selected_category}")
                    return [selected_category]
                
                elif choice_num == len(categories) + 1:
                    print("\nSelected: All categories")
                    return categories
                
                elif choice_num == len(categories) + 2:
                    return self.get_multiple_categories_choice(categories)
                
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(categories) + 2}")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_multiple_categories_choice(self, categories):
        """Allow user to select multiple categories"""
        print("\n" + "-"*40)
        print("MULTIPLE CATEGORY SELECTION")
        print("-"*40)
        print("Enter the numbers of categories you want to split (comma-separated)")
        print("Example: 1,3 for Animation and Natural Content")
        
        for i, category in enumerate(categories, 1):
            print(f"  {i}. {category}")
        
        while True:
            try:
                choices = input("\nEnter your choices (e.g., 1,2,4): ").strip()
                
                if not choices:
                    print("No selection made. Please enter at least one number.")
                    continue
                
                choice_nums = [int(x.strip()) for x in choices.split(',')]
                
                invalid_choices = [num for num in choice_nums if num < 1 or num > len(categories)]
                if invalid_choices:
                    print(f"Invalid choices: {invalid_choices}. Please use numbers between 1 and {len(categories)}")
                    continue
                
                selected_categories = [categories[num - 1] for num in choice_nums]
                selected_categories = list(set(selected_categories))
                
                print(f"\nSelected categories: {', '.join(selected_categories)}")
                
                confirm = input("Proceed with this selection? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return selected_categories
                else:
                    print("Please make your selection again.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas (e.g., 1,2,3)")

    def show_category_preview(self, categories_to_split):
        """Show preview of what will be split"""
        print("\n" + "="*50)
        print("SPLIT PREVIEW")
        print("="*50)
        print(f"Split ratios: Train={self.train_ratio}, Val={self.val_ratio}, Test={self.test_ratio}")
        print(f"Source directory: {self.source_base_dir}")
        print(f"Target directory: {self.target_dir}")
        print(f"Random seed: {self.seed}")
        
        total_to_process = 0
        total_already_processed = 0
        
        for category in categories_to_split:
            print(f"\n{category}:")
            subcategories = self.category_structure[category]
            for subcategory in subcategories:
                subcategory_path = self.source_base_dir / category / "videos" / subcategory
                if subcategory_path.exists():
                    videos = []
                    for ext in self.video_extensions:
                        videos.extend(list(subcategory_path.glob(ext)))
                    videos = list(set(videos))
                    
                    # Check how many are already processed
                    already_processed = sum(1 for v in videos if self.is_file_already_processed(v))
                    to_process = len(videos) - already_processed
                    
                    total_to_process += to_process
                    total_already_processed += already_processed
                    
                    if already_processed > 0:
                        print(f"  - {subcategory}: {len(videos)} videos ({to_process} new, {already_processed} already split)")
                    else:
                        print(f"  - {subcategory}: {len(videos)} videos")
                else:
                    print(f"  - {subcategory}: [NOT FOUND]")
        
        print("\n" + "="*50)
        if total_already_processed > 0:
            print(f"Total: {total_to_process} new files to process, {total_already_processed} already split")
        print("="*50)
        
        confirm = input("Proceed with splitting? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']

    def split_videos_by_category(self, categories_to_split=None):
        """Split videos and preserve subcategory structure inside train/val/test."""
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

                # Filter out already processed files
                unprocessed_videos = [v for v in videos if not self.is_file_already_processed(v)]
                already_processed = len(videos) - len(unprocessed_videos)

                if already_processed > 0:
                    print(f"   Found {len(videos)} videos in {subcategory} ({already_processed} already processed)")
                else:
                    print(f"   Found {len(videos)} videos in {subcategory}")

                if not unprocessed_videos:
                    print(f"   All files already processed, skipping...")
                    continue

                # Shuffle only unprocessed videos
                random.shuffle(unprocessed_videos)

                # Split train/val/test counts
                total = len(unprocessed_videos)
                train_count = int(total * self.train_ratio)
                val_count = int(total * self.val_ratio)
                test_count = total - train_count - val_count

                train_files = unprocessed_videos[:train_count]
                val_files = unprocessed_videos[train_count:train_count + val_count]
                test_files = unprocessed_videos[train_count + val_count:]

                print(f"     Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)} files")

                # Copy to subcategory-preserving structure
                self._copy_files(train_files, 'train', standard_category, subcategory)
                self._copy_files(val_files, 'val', standard_category, subcategory)
                self._copy_files(test_files, 'test', standard_category, subcategory)

        # Save manifest after all processing
        self.save_processed_files()

    def _copy_files(self, files, split, main_category, subcategory):
        """Copy to target/raw/{split}/{main_category}/{subcategory}"""
        if not files:
            return
            
        target_path = self.target_dir / "raw" / split / main_category / subcategory
        target_path.mkdir(parents=True, exist_ok=True)

        print(f"     Copying {len(files)} files → {target_path}")

        copied_count = 0
        skipped_count = 0

        for i, file_path in enumerate(files):
            dest_path = target_path / file_path.name
            
            # Check if already exists in ANY split (additional safety)
            if self.is_file_already_processed(file_path):
                existing_split = self.processed_files[file_path.name]['split']
                if existing_split != split:
                    print(f"       WARNING: {file_path.name} already in {existing_split} split, skipping!")
                skipped_count += 1
                continue
                
            if dest_path.exists():
                print(f"       WARNING: {dest_path} already exists, skipping!")
                skipped_count += 1
                continue
                
            try:
                shutil.copy2(file_path, dest_path)
                self.mark_file_as_processed(file_path, split, main_category, subcategory)
                copied_count += 1
                
                if (copied_count) % 50 == 0:
                    print(f"       Copied {copied_count}/{len(files)} files")
            except Exception as e:
                print(f"       Error copying {file_path}: {e}")

        print(f"     ✓ Copied: {copied_count}, Skipped: {skipped_count}")

    def verify_split(self):
        """Verify file counts after splitting"""
        print("\n=== SPLIT VERIFICATION ===")
        print(f"Using random seed: {self.seed}")

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
                if main_count > 0:
                    print(f"   {main_cat}: {main_count} files")

                    for subcat in cat_path.iterdir():
                        if subcat.is_dir():
                            sub_count = sum(len(list(subcat.glob(ext))) for ext in self.video_extensions)
                            if sub_count > 0:
                                print(f"      {subcat.name}: {sub_count}")


def main():
    source_directory = "Dataset"
    target_directory = "video_classification_project/data"

    print("Starting interactive dataset splitter...")
    splitter = VideoDatasetSplitter(source_directory, target_directory, 0.7, 0.2, 0.1, seed=42)

    # Check for existing splits
    if not splitter.check_existing_splits():
        return

    # Get user's category choice
    categories_to_split = splitter.get_user_category_choice()
    
    if categories_to_split is None:
        print("No categories selected. Exiting...")
        return
    
    # Show preview and get confirmation
    if not splitter.show_category_preview(categories_to_split):
        print("Operation cancelled by user.")
        return
    
    # Perform the split
    print("\n" + "="*50)
    print("STARTING DATASET SPLIT...")
    print("="*50)
    
    splitter.split_videos_by_category(categories_to_split=categories_to_split)
    splitter.verify_split()
    
    print("\n" + "="*50)
    print("DATASET SPLIT COMPLETED!")
    print("="*50)
    print(f"\nManifest saved at: {splitter.split_manifest_path}")
    print("This manifest tracks all processed files to prevent duplicates.")


if __name__ == "__main__":
    main()