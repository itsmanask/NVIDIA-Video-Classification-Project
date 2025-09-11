import os
import shutil
import random
from pathlib import Path


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
                          'Dragon Ball', 'Sonic the Hedgehog', 'One Piece', 'Walt Disney World',
                          'Bleach', 'Mickey Mouse'],

            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty','Grand Theft Auto', 'Grand Theft Auto V',
                       'World of Warcraft', 'League of Legends', 'Battlefield', 'RuneScape',
                       'Action-adventure game', 'FIFA 15', 'Counter-Strike', 'Need for Speed'],

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

    def get_user_category_choice(self):
        """
        Display menu and get user's category selection
        """
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
                    # Single category selected
                    selected_category = categories[choice_num - 1]
                    print(f"\nSelected: {selected_category}")
                    return [selected_category]
                
                elif choice_num == len(categories) + 1:
                    # All categories
                    print("\nSelected: All categories")
                    return categories
                
                elif choice_num == len(categories) + 2:
                    # Multiple categories (custom selection)
                    return self.get_multiple_categories_choice(categories)
                
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(categories) + 2}")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_multiple_categories_choice(self, categories):
        """
        Allow user to select multiple categories
        """
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
                
                # Parse comma-separated choices
                choice_nums = [int(x.strip()) for x in choices.split(',')]
                
                # Validate choices
                invalid_choices = [num for num in choice_nums if num < 1 or num > len(categories)]
                if invalid_choices:
                    print(f"Invalid choices: {invalid_choices}. Please use numbers between 1 and {len(categories)}")
                    continue
                
                # Get selected categories
                selected_categories = [categories[num - 1] for num in choice_nums]
                selected_categories = list(set(selected_categories))  # Remove duplicates
                
                print(f"\nSelected categories: {', '.join(selected_categories)}")
                
                # Confirm selection
                confirm = input("Proceed with this selection? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return selected_categories
                else:
                    print("Please make your selection again.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas (e.g., 1,2,3)")

    def show_category_preview(self, categories_to_split):
        """
        Show preview of what will be split
        """
        print("\n" + "="*50)
        print("SPLIT PREVIEW")
        print("="*50)
        print(f"Split ratios: Train={self.train_ratio}, Val={self.val_ratio}, Test={self.test_ratio}")
        print(f"Source directory: {self.source_base_dir}")
        print(f"Target directory: {self.target_dir}")
        print(f"Random seed: {self.seed}")
        
        for category in categories_to_split:
            print(f"\n{category}:")
            subcategories = self.category_structure[category]
            for subcategory in subcategories:
                subcategory_path = self.source_base_dir / category / "videos" / subcategory
                if subcategory_path.exists():
                    # Count videos
                    videos = []
                    for ext in self.video_extensions:
                        videos.extend(list(subcategory_path.glob(ext)))
                    video_count = len(set(videos))
                    print(f"  - {subcategory}: {video_count} videos")
                else:
                    print(f"  - {subcategory}: [NOT FOUND]")
        
        print("\n" + "="*50)
        confirm = input("Proceed with splitting? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']

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

                # Shuffle for randomness (using the set seed)
                random.shuffle(videos)

                # Split train/val/test counts
                total = len(videos)
                train_count = int(total * self.train_ratio)
                val_count = int(total * self.val_ratio)
                test_count = total - train_count - val_count

                train_files = videos[:train_count]
                val_files = videos[train_count:train_count + val_count]
                test_files = videos[train_count + val_count:]

                print(f"     Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)} files")

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


# ------------------------- Usage Example -------------------------
def main():
    source_directory = "Dataset"
    target_directory = "video_classification_project/data"

    print("Starting interactive dataset splitter...")
    # You can change the seed value here for different random splits
    # Using seed=42 ensures reproducible results
    splitter = VideoDatasetSplitter(source_directory, target_directory, 0.7, 0.2, 0.1, seed=42)

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


if __name__ == "__main__":
    main()