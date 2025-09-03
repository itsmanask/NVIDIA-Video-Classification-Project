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
    def __init__(self, input_dir, output_dir, frames_per_video=30, img_size=(224, 224)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video at regular intervals"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        
        if total_frames == 0:
            cap.release()
            return frames
        
        # Calculate frame indices to extract
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            # Extract frames at regular intervals
            step = total_frames // max_frames
            frame_indices = [i * step for i in range(max_frames)]
        
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
        """Preprocess a single video"""
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
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.Resize(self.img_size),
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
            # Pad with zeros
            padding_needed = self.frames_per_video - len(processed_frames)
            zero_frame = torch.zeros_like(processed_frames[0])
            processed_frames.extend([zero_frame] * padding_needed)
        elif len(processed_frames) > self.frames_per_video:
            processed_frames = processed_frames[:self.frames_per_video]
        
        # Stack frames into tensor [frames, channels, height, width]
        video_tensor = torch.stack(processed_frames)
        
        return video_tensor
    
    def process_split(self, split_name, augment=False):
        """Process all videos in a split (train/val/test)"""
        split_input_dir = self.input_dir / split_name
        split_output_dir = self.output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category mapping
        category_to_idx = {
            'Animation': 0,
            'Gaming': 1, 
            'Natural_Content': 2,
            'Flat_Content': 3
        }
        
        processed_data = []
        labels = []
        filenames = []
        
        print(f"\nProcessing {split_name} split...")
        
        # Process each category
        for category in category_to_idx.keys():
            category_path = split_input_dir / category
            
            if not category_path.exists():
                print(f"Category {category} not found in {split_name}")
                continue
            
            category_label = category_to_idx[category]
            
            # Get all video files in category (including subcategories)
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(list(category_path.rglob(ext)))
            
            print(f"  Processing {len(video_files)} videos in {category}...")
            
            for video_path in tqdm(video_files, desc=f"{category}"):
                try:
                    # Preprocess video
                    video_tensor = self.preprocess_video(video_path, augment=augment)
                    
                    if video_tensor is not None:
                        processed_data.append(video_tensor)
                        labels.append(category_label)
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
                'category_mapping': category_to_idx
            }
            
            output_path = split_output_dir / 'processed_data.pt'
            torch.save(data_dict, output_path)
            
            print(f"Saved {len(processed_data)} processed videos to {output_path}")
            
            # Save metadata
            metadata = {
                'num_videos': len(processed_data),
                'frames_per_video': self.frames_per_video,
                'image_size': self.img_size,
                'category_mapping': category_to_idx,
                'augmented': augment
            }
            
            with open(split_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return len(processed_data)
    
    def process_all_splits(self):
        """Process train, validation, and test splits"""
        print("Starting video preprocessing...")
        
        # Process each split
        train_count = self.process_split('train', augment=True)
        val_count = self.process_split('val', augment=False)
        test_count = self.process_split('test', augment=False)
        
        print(f"\n=== PREPROCESSING COMPLETE ===")
        print(f"Training videos processed: {train_count}")
        print(f"Validation videos processed: {val_count}")
        print(f"Test videos processed: {test_count}")
        print(f"Total: {train_count + val_count + test_count}")

# Usage example
def main():
    preprocessor = VideoPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed", 
        frames_per_video=30,  # Extract 30 frames per video
        img_size=(224, 224)   # Resize to 224x224 for transfer learning
    )
    
    preprocessor.process_all_splits()

if __name__ == "__main__":
    main()