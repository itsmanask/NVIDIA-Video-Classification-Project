import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm


class VideoPreprocessor:
    def __init__(self, input_dir, output_dir, frames_per_video=32, img_size=(224, 224), clip_duration=10):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.sampling_strategy = 'uniform'
        self.clip_duration = clip_duration
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.category_structure = {
            'Animation': ['Cartoon', 'Animation', 'Lego minifigure', 'Naruto', 'The Walt Disney Company',
                          'Dragon Ball', 'Sonic the Hedgehog', 'One Piece', 'Walt Disney World',
                          'Bleach', 'Mickey Mouse'],
            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty', 'Grand Theft Auto V',
                      'World of Warcraft', 'Call of Duty: Black Ops II', 'League of Legends',
                      'Battlefield', 'Grand Theft Auto: San Andreas', 'RuneScape',
                      'Call of Duty: Modern Warfare 3', 'Call of Duty: Black Ops', 'FIFA 15',
                      'Counter-Strike', 'Need for Speed'],
            'Natural_Content': ['Animal', 'Pet', 'Fishing', 'Fish', 'Outdoor recreation', 'Dog',
                               'Horse', 'Bird', 'Plant', 'Cat', 'Farm', 'Garden', 'Nature',
                               'Tree', 'Wildlife', 'Chicken', 'Lion', 'Deer', 'Bear', 'Elephant'],
            'Flat_Content': ['Website', 'Chart', 'Map', 'Logo', 'Text', 'Typography',
                             'Screencast', 'Illustration', 'Poster']
        }
        self.all_categories = {cat: idx for idx, cat in enumerate(self.category_structure.keys())}

    def get_user_category_selection(self):
        """Interactive category selection with sampling strategy options"""
        print("\n" + "=" * 50)
        print("VIDEO PREPROCESSING - CATEGORY SELECTION")
        print("=" * 50)
        print("Available categories:")
        categories_list = list(self.all_categories.keys())
        for i, category in enumerate(categories_list, 1):
            print(f"  {i}. {category}")
        print(f"  {len(categories_list) + 1}. All categories")
        print("  0. Exit")
        while True:
            try:
                print(f"\nSelect categories to process:")
                print("- Enter single number (e.g., '1' for Animation)")
                print("- Enter multiple numbers separated by commas (e.g., '1,3' for Animation and Natural_Content)")
                print(f"- Enter '{len(categories_list) + 1}' for all categories")
                print("- Enter '0' to exit")
                user_input = input("\nYour selection: ").strip()
                if user_input == '0':
                    print("Exiting...")
                    return None, None
                selections = [int(x.strip()) for x in user_input.split(',')]
                valid_range = list(range(1, len(categories_list) + 2))
                if not all(s in valid_range for s in selections):
                    print(f"Invalid selection. Please enter numbers between 1-{len(categories_list) + 1} or 0 to exit.")
                    continue
                if len(categories_list) + 1 in selections:
                    selected_categories = self.all_categories.copy()
                    print(f"\nSelected: All categories")
                else:
                    selected_categories = {}
                    selected_names = []
                    for selection in selections:
                        category_name = categories_list[selection - 1]
                        selected_categories[category_name] = self.all_categories[category_name]
                        selected_names.append(category_name)
                    print(f"\nSelected categories: {', '.join(selected_names)}")
                break
            except ValueError:
                print("Invalid input. Please enter numbers only.")
            except IndexError:
                print(f"Invalid selection. Please enter numbers between 1-{len(categories_list) + 1} or 0 to exit.")

        print(f"\n" + "-" * 50)
        print("SAMPLING STRATEGY (for 2-minute YouTube videos):")
        print("1. Uniform - Sample frames across entire video (2 minutes)")
        print("2. Middle Clip - Focus on middle 10 seconds (most content-rich)")
        print("3. Random Clip - Random 10-second clips (training diversity)")
        while True:
            try:
                strategy_input = input("\nSelect sampling strategy (1-3): ").strip()
                if strategy_input == '1':
                    sampling_strategy = 'uniform'
                    print("Selected: Uniform sampling across entire video")
                    break
                elif strategy_input == '2':
                    sampling_strategy = 'middle_clip'
                    print("Selected: Middle 10-second clip sampling")
                    break
                elif strategy_input == '3':
                    sampling_strategy = 'random_clip'
                    print("Selected: Random 10-second clip sampling")
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter 1, 2, or 3")
        self.sampling_strategy = sampling_strategy
        print(f"\nProcessing {len(selected_categories)} categories with {sampling_strategy} sampling:")
        for category in selected_categories.keys():
            print(f"  - {category}")
        confirm = input("\nProceed with preprocessing? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Preprocessing cancelled.")
            return None, None
        return selected_categories, sampling_strategy

    def extract_frames(self, video_path, max_frames=32):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        frames = []
        if total_frames == 0 or duration < 1.0:
            cap.release()
            return frames
        if self.sampling_strategy == 'middle_clip':
            clip_frames = min(int(self.clip_duration * fps), total_frames)
            start_frame = max(0, (total_frames - clip_frames) // 2)
            end_frame = start_frame + clip_frames
        elif self.sampling_strategy == 'random_clip' and hasattr(self, '_training_mode') and self._training_mode:
            clip_frames = min(int(self.clip_duration * fps), total_frames)
            max_start = max(0, total_frames - clip_frames)
            start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            end_frame = start_frame + clip_frames
        else:
            start_frame = 0
            end_frame = total_frames
        sampling_frames = end_frame - start_frame
        if sampling_frames <= max_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = sampling_frames / max_frames
            frame_indices = [start_frame + int(i * step) for i in range(max_frames)]
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        cap.release()
        return frames

    def preprocess_video(self, video_path, augment=False):
        self._training_mode = augment
        frames = self.extract_frames(video_path, self.frames_per_video)
        if len(frames) == 0:
            return None
        processed_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            if augment:
                augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomResizedCrop(self.img_size[0], scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                processed_frame = augment_transform(pil_frame)
            else:
                processed_frame = self.transform(pil_frame)
            processed_frames.append(processed_frame)
        if len(processed_frames) < self.frames_per_video:
            padding_needed = self.frames_per_video - len(processed_frames)
            zero_frame = torch.zeros_like(processed_frames[0])
            processed_frames.extend([zero_frame] * padding_needed)
        elif len(processed_frames) > self.frames_per_video:
            processed_frames = processed_frames[:self.frames_per_video]
        video_tensor = torch.stack(processed_frames)
        return video_tensor

    def process_split(self, split_name, selected_categories, augment=False):
        split_input_dir = self.input_dir / split_name
        split_output_dir = self.output_dir / split_name
        print(f"\nProcessing {split_name} split...")

        for category, category_idx in selected_categories.items():
            subcategories = self.category_structure[category]
            category_output_dir = split_output_dir / category
            category_output_dir.mkdir(parents=True, exist_ok=True)

            for subcat in subcategories:
                subcat_input_dir = split_input_dir / category / 'videos' / subcat
                subcat_output_dir = category_output_dir / subcat
                subcat_output_dir.mkdir(parents=True, exist_ok=True)

                if not subcat_input_dir.exists():
                    print(f"Subcategory path {subcat_input_dir} not found in {split_name}")
                    continue

                video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                video_files = [file for file in subcat_input_dir.iterdir()
                               if file.is_file() and file.suffix.lower() in video_extensions]
                print(f"  Processing {len(video_files)} videos in {category}/{subcat}...")

                processed_data = []
                labels = []
                filenames = []

                for video_path in tqdm(video_files, desc=f"{category}-{subcat}"):
                    try:
                        video_tensor = self.preprocess_video(video_path, augment=augment)
                        if video_tensor is not None:
                            processed_data.append(video_tensor)
                            labels.append(category_idx)
                            filenames.append(video_path.name)
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        continue

                if processed_data:
                    data_dict = {
                        'videos': torch.stack(processed_data),
                        'labels': torch.tensor(labels),
                        'filenames': filenames,
                        'category_mapping': {category: category_idx}
                    }
                    output_path = subcat_output_dir / 'processed_data.pt'
                    torch.save(data_dict, output_path)
                    print(f"Saved {len(processed_data)} processed videos to {output_path}")

                    metadata = {
                        'num_videos': len(processed_data),
                        'frames_per_video': self.frames_per_video,
                        'image_size': self.img_size,
                        'category_mapping': {category: category_idx},
                        'augmented': augment,
                        'selected_category': category,
                        'selected_subcategory': subcat
                    }
                    with open(subcat_output_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)

    def process_all_splits(self, selected_categories):
        print(f"\nStarting video preprocessing for {len(selected_categories)} categories...")
        print(f"Sampling strategy: {self.sampling_strategy}")
        print(f"Frames per video: {self.frames_per_video}")
        for split in ['train', 'val', 'test']:
            augment = (split == 'train')
            self.process_split(split, selected_categories, augment=augment)
        print(f"\n=== PREPROCESSING COMPLETE ===")


def main():
    preprocessor = VideoPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed",
        frames_per_video=32,
        img_size=(224, 224),
        clip_duration=10
    )
    selected_categories, sampling_strategy = preprocessor.get_user_category_selection()
    if selected_categories is not None:
        preprocessor.process_all_splits(selected_categories)
    else:
        print("No categories selected. Exiting.")


if __name__ == "__main__":
    main()
