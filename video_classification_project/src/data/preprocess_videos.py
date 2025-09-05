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
        self.clip_duration = clip_duration
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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
        self.all_categories = list(self.category_structure.keys())
        self.processed_categories = set()  # Track completed categories

    def get_sampling_strategy_for_subcategory(self, category, subcategory):
        print(f"\nSelect sampling strategy for subcategory '{subcategory}' under '{category}':")
        print("1. Uniform - Sample frames across entire video (2 minutes)")
        print("2. Middle Clip - Focus on middle 10 seconds (most content-rich)")
        print("3. Random Clip - Random 10-second clips (training diversity)")
        while True:
            strategy_input = input(f"Sampling strategy (1-3) for {subcategory}: ").strip()
            if strategy_input in ['1', '2', '3']:
                if strategy_input == '1':
                    return 'uniform'
                elif strategy_input == '2':
                    return 'middle_clip'
                else:
                    return 'random_clip'
            else:
                print("Invalid input. Please enter 1, 2, or 3.")

    def extract_frames(self, video_path, max_frames=32, sampling_strategy='uniform'):
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
        if sampling_strategy == 'middle_clip':
            clip_frames = min(int(self.clip_duration * fps), total_frames)
            start_frame = max(0, (total_frames - clip_frames) // 2)
            end_frame = start_frame + clip_frames
        elif sampling_strategy == 'random_clip':
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

    def preprocess_video(self, video_path, augment=False, sampling_strategy='uniform'):
        frames = self.extract_frames(video_path, self.frames_per_video, sampling_strategy)
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
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
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

    def process_subcategory(self, split_name, category, category_idx, subcategory, augment, sampling_strategy):
        split_input_dir = self.input_dir / split_name
        split_output_dir = self.output_dir / split_name / category / subcategory
        split_output_dir.mkdir(parents=True, exist_ok=True)

        subcat_input_dir = split_input_dir / category / 'videos' / subcategory
        if not subcat_input_dir.exists():
            print(f"Subcategory path {subcat_input_dir} not found in {split_name}")
            return 0

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [file for file in subcat_input_dir.iterdir()
                       if file.is_file() and file.suffix.lower() in video_extensions]
        print(f"  Processing {len(video_files)} videos in {category}/{subcategory} with '{sampling_strategy}' strategy...")

        processed_data = []
        labels = []
        filenames = []

        for video_path in tqdm(video_files, desc=f"{category}-{subcategory}"):
            try:
                video_tensor = self.preprocess_video(video_path, augment=augment, sampling_strategy=sampling_strategy)
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
            output_path = split_output_dir / 'processed_data.pt'
            torch.save(data_dict, output_path)
            print(f"Saved {len(processed_data)} processed videos to {output_path}")

            metadata = {
                'num_videos': len(processed_data),
                'frames_per_video': self.frames_per_video,
                'image_size': self.img_size,
                'category_mapping': {category: category_idx},
                'augmented': augment,
                'selected_category': category,
                'selected_subcategory': subcategory,
                'sampling_strategy': sampling_strategy
            }
            with open(split_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            return len(processed_data)
        else:
            print(f"No videos processed in {category}/{subcategory} for split {split_name}.")
            return 0

    def interactive_process(self):
        all_processed_categories = set()
        while True:
            print("\nAvailable main categories:")
            for i, cat in enumerate(self.all_categories, 1):
                print(f"  {i}. {cat}{' (processed)' if cat in all_processed_categories else ''}")
            print("  0. Exit")

            try:
                choice = int(input("\nSelect a main category to process (or 0 to exit): ").strip())
            except ValueError:
                print("Invalid input. Enter a number.")
                continue

            if choice == 0:
                print("Exiting preprocessing.")
                break

            if not 1 <= choice <= len(self.all_categories):
                print(f"Enter a number between 1 and {len(self.all_categories)}")
                continue

            selected_category = self.all_categories[choice - 1]
            if selected_category in all_processed_categories:
                print(f"Category '{selected_category}' has already been processed.")
                continue

            category_idx = self.all_categories.index(selected_category)
            subcategories = self.category_structure[selected_category]

            for subcat in subcategories:
                sampling_strategy = self.get_sampling_strategy_for_subcategory(selected_category, subcat)
                # Process each split for this subcategory
                for split in ['train', 'val', 'test']:
                    augment = (split == 'train')
                    self.process_subcategory(split, selected_category, category_idx, subcat, augment, sampling_strategy)

            all_processed_categories.add(selected_category)
            print(f"Completed processing for category '{selected_category}'.")

        print("\nAll selected processing done. Goodbye!")


def main():
    preprocessor = VideoPreprocessor(
        input_dir="data/raw",           # Your raw videos root folder with train/test/val splits
        output_dir="data/processed",
        frames_per_video=32,
        img_size=(224, 224),
        clip_duration=10
    )
    preprocessor.interactive_process()


if __name__ == "__main__":
    main()
