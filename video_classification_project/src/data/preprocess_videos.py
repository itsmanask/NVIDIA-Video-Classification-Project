import cv2
import os
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
        self.frames_per_video = frames_per_video  # Optimized for 2-min YouTube videos
        self.img_size = img_size
        self.sampling_strategy = 'uniform'  # Default, will be set by user selection
        self.clip_duration = clip_duration  # Focus on N seconds of the video
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Define available categories (matching your dataset structure)
        self.all_categories = {
            'Animation': 0,
            'Gaming': 1, 
            'Natural_Content': 2,  # Note: keeping underscore for consistency
            'Flat_Content': 3      # Note: keeping underscore for consistency
        }
    
    def get_user_category_selection(self):
        """Interactive category selection with sampling strategy options"""
        print("\n" + "="*50)
        print("VIDEO PREPROCESSING - CATEGORY SELECTION")
        print("="*50)
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
                
                # Parse input
                selections = [int(x.strip()) for x in user_input.split(',')]
                
                # Validate selections
                valid_range = list(range(1, len(categories_list) + 2))
                if not all(s in valid_range for s in selections):
                    print(f"Invalid selection. Please enter numbers between 1-{len(categories_list) + 1} or 0 to exit.")
                    continue
                
                # Handle "all categories" selection
                if len(categories_list) + 1 in selections:
                    selected_categories = self.all_categories.copy()
                    print(f"\nSelected: All categories")
                    break
                
                # Handle specific category selections
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
        
        # Sampling strategy selection for YouTube videos
        print(f"\n" + "-"*50)
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
        
        # Update sampling strategy
        self.sampling_strategy = sampling_strategy
        
        # Confirm selection
        print(f"\nProcessing {len(selected_categories)} categories with {sampling_strategy} sampling:")
        for category in selected_categories.keys():
            print(f"  - {category}")
        
        confirm = input("\nProceed with preprocessing? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Preprocessing cancelled.")
            return None, None
        
        return selected_categories, sampling_strategy
    
    def extract_frames(self, video_path, max_frames=32):
        """Extract frames optimized for 2-minute YouTube videos"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        
        if total_frames == 0 or duration < 1.0:  # Skip very short videos
            cap.release()
            return frames
        
        # Determine sampling region based on strategy
        if self.sampling_strategy == 'middle_clip':
            # Sample from middle portion (often most content-rich)
            clip_frames = min(int(self.clip_duration * fps), total_frames)
            start_frame = max(0, (total_frames - clip_frames) // 2)
            end_frame = start_frame + clip_frames
            
        elif self.sampling_strategy == 'random_clip' and hasattr(self, '_training_mode') and self._training_mode:
            # Random clip for training diversity
            clip_frames = min(int(self.clip_duration * fps), total_frames)
            max_start = max(0, total_frames - clip_frames)
            start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            end_frame = start_frame + clip_frames
            
        else:  # uniform sampling across entire video
            start_frame = 0
            end_frame = total_frames
        
        sampling_frames = end_frame - start_frame
        
        # Calculate frame indices to extract
        if sampling_frames <= max_frames:
            # Take all frames in the sampling region
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Uniform sampling within the region
            step = sampling_frames / max_frames
            frame_indices = [start_frame + int(i * step) for i in range(max_frames)]
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            if len(frames) >= max_frames:
                break
        
        cap.release()
        return frames
    
    def preprocess_video(self, video_path, augment=False):
        """Preprocess a single video with enhanced frame sampling"""
        # Set training mode flag for frame sampling
        self._training_mode = augment
        
        frames = self.extract_frames(video_path, self.frames_per_video)
        
        if len(frames) == 0:
            return None
        
        # Convert frames to tensors
        processed_frames = []
        
        for frame in frames:
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame)
            
            # Apply data augmentation if specified
            if augment:
                augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(15),  # Increased rotation for robustness
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomResizedCrop(self.img_size[0], scale=(0.8, 1.0)),  # Add scale augmentation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                processed_frame = augment_transform(pil_frame)
            else:
                processed_frame = self.transform(pil_frame)
            
            processed_frames.append(processed_frame)
        
        # Pad or truncate to fixed length
        if len(processed_frames) < self.frames_per_video:
            # Pad with zeros (representing empty frames)
            padding_needed = self.frames_per_video - len(processed_frames)
            zero_frame = torch.zeros_like(processed_frames[0])
            processed_frames.extend([zero_frame] * padding_needed)
        elif len(processed_frames) > self.frames_per_video:
            processed_frames = processed_frames[:self.frames_per_video]
        
        # Stack frames into tensor [frames, channels, height, width]
        video_tensor = torch.stack(processed_frames)
        
        return video_tensor
    
    def process_split(self, split_name, selected_categories, augment=False):
        """Process all videos in a split (train/val/test) for selected categories only"""
        split_input_dir = self.input_dir / split_name
        split_output_dir = self.output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        labels = []
        filenames = []
        
        print(f"\nProcessing {split_name} split...")
        
        # Process only selected categories
        for category, category_idx in selected_categories.items():
            # Update path to match actual dataset structure: Category/videos/Subcategory/
            category_videos_path = split_input_dir / category / 'videos'
            
            if not category_videos_path.exists():
                print(f"Category videos path {category_videos_path} not found in {split_name}")
                continue
            
            # Get all video files from all subcategories within this category
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            # Iterate through all subcategories in the category/videos/ folder
            for subcategory_dir in category_videos_path.iterdir():
                if subcategory_dir.is_dir():
                    # Get all video files in this subcategory (case-insensitive)
                    for file in subcategory_dir.iterdir():
                        if file.is_file() and file.suffix.lower() in video_extensions:
                            video_files.append(file)
            
            print(f"  Processing {len(video_files)} videos in {category}...")
            
            for video_path in tqdm(video_files, desc=f"{category}"):
                try:
                    # Preprocess video
                    video_tensor = self.preprocess_video(video_path, augment=augment)
                    
                    if video_tensor is not None:
                        processed_data.append(video_tensor)
                        labels.append(category_idx)
                        filenames.append(video_path.name)
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue
        
        # Save processed data
        if processed_data:
            data_dict = {
                'videos': torch.stack(processed_data),  # [N, frames, channels, height, width]
                'labels': torch.tensor(labels),         # [N]
                'filenames': filenames,
                'category_mapping': selected_categories
            }
            
            output_path = split_output_dir / 'processed_data.pt'
            torch.save(data_dict, output_path)
            
            print(f"Saved {len(processed_data)} processed videos to {output_path}")
            
            # Save metadata
            metadata = {
                'num_videos': len(processed_data),
                'frames_per_video': self.frames_per_video,
                'image_size': self.img_size,
                'category_mapping': selected_categories,
                'augmented': augment,
                'selected_categories': list(selected_categories.keys())
            }
            
            with open(split_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return len(processed_data)
    
    def process_all_splits(self, selected_categories):
        """Process train, validation, and test splits for selected categories"""
        print(f"\nStarting video preprocessing for {len(selected_categories)} categories...")
        print(f"Sampling strategy: {self.sampling_strategy}")
        print(f"Frames per video: {self.frames_per_video}")
        
        # Process each split
        train_count = self.process_split('train', selected_categories, augment=True)
        val_count = self.process_split('val', selected_categories, augment=False)
        test_count = self.process_split('test', selected_categories, augment=False)
        
        print(f"\n=== PREPROCESSING COMPLETE ===")
        print(f"Selected categories: {', '.join(selected_categories.keys())}")
        print(f"Sampling strategy: {self.sampling_strategy}")
        print(f"Training videos processed: {train_count}")
        print(f"Validation videos processed: {val_count}")
        print(f"Test videos processed: {test_count}")
        print(f"Total: {train_count + val_count + test_count}")
        print(f"Average video duration: ~2 minutes (YouTube 8M dataset)")
        if self.sampling_strategy != 'uniform':
            print(f"Focused on: {self.clip_duration}-second clips per video")

# Usage example
def main():
    preprocessor = VideoPreprocessor(
        input_dir="Dataset",  # Updated to match your dataset structure
        output_dir="data/processed", 
        frames_per_video=32,         # Optimal for 2-minute YouTube videos
        img_size=(224, 224),         # Standard for transfer learning
        clip_duration=10             # Focus on 10-second clips
    )
    
    # Get user category and strategy selection
    selected_categories, sampling_strategy = preprocessor.get_user_category_selection()
    
    if selected_categories is not None:
        preprocessor.process_all_splits(selected_categories)
    else:
        print("No categories selected. Exiting.")

if __name__ == "__main__":
    main()