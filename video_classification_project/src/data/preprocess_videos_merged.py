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
        
        # Initialize processing mode
        self.processing_mode = None
        self.device = None
        self.batch_size = 1  # Default for CPU
        
        # Basic CPU transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # GPU-specific attributes (will be set only if GPU mode is chosen)
        self.normalize_mean = None
        self.normalize_std = None
        
        self.category_structure = {
            'Animation': ['Cartoon', 'Animation', 'Lego minifigure', 'Naruto',
                          'The Walt Disney Company', 'Dragon Ball', 'Sonic the Hedgehog',
                          'One Piece', 'Walt Disney World', 'Bleach', 'Mickey Mouse'],
            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty', 'Grand Theft Auto V',
                       'World of Warcraft', 'League of Legends', 'Battlefield', 'RuneScape',
                       'Action-adventure game', 'FIFA 15', 'Counter-Strike', 'Need for Speed'],
            'Natural_Content': ['Animal', 'Pet', 'Fishing', 'Fish', 'Outdoor recreation', 'Dog',
                                'Horse', 'Bird', 'Plant', 'Cat', 'Farm', 'Garden', 'Nature',
                                'Tree', 'Wildlife', 'Chicken', 'Lion', 'Deer', 'Bear', 'Elephant'],
            'Flat_Content': ['Website', 'Chart', 'Map', 'Logo', 'Text', 'Typography',
                             'Screencast', 'Illustration', 'Poster']
        }
        self.all_categories = list(self.category_structure.keys())
        self.resume_file = self.output_dir / "resume_checkpoint.json"

    def setup_processing_mode(self):
        """Setup CPU or GPU processing mode based on user choice"""
        print("\n" + "="*50)
        print("VIDEO PREPROCESSOR - PROCESSING MODE SELECTION")
        print("="*50)
        
        print(f"\nProcessing Mode Options:")
        print(f"1. CPU Mode - Compatible with all systems (slower but reliable)")
        print(f"2. GPU Mode - Faster processing (requires CUDA-compatible GPU)")
        
        while True:
            try:
                choice = int(input(f"\nSelect processing mode (1 for CPU, 2 for GPU): ").strip())
                if choice == 1:
                    self.processing_mode = 'cpu'
                    self.device = torch.device('cpu')
                    self.batch_size = 1
                    print(f"Selected: CPU Mode")
                    break
                elif choice == 2:
                    # Check GPU availability only when GPU mode is selected
                    if torch.cuda.is_available():
                        self.processing_mode = 'gpu'
                        self.device = torch.device('cuda')
                        # Setup GPU-specific components
                        self._setup_gpu_components()
                        print(f"Selected: GPU Mode")
                        print(f"GPU: {torch.cuda.get_device_name()}")
                        break
                    else:
                        print("GPU not available on this system. Please select CPU mode (1).")
                        continue
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        print(f"Using device: {self.device}")
        print("="*50 + "\n")

    def _setup_gpu_components(self):
        """Setup GPU-specific components (only called when GPU mode is selected)"""
        # GPU batch processing transforms (without normalization)
        self.transform_gpu = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        
        # GPU normalization tensors
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(3, 1, 1)
        
        # Determine optimal batch size for GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        self.batch_size = min(16, max(4, int(gpu_memory / 4e9)))  # Conservative estimate
        print(f"GPU batch size: {self.batch_size}")

    def normalize_on_gpu(self, tensor_batch):
        """Perform normalization on GPU for better performance (GPU mode only)"""
        return (tensor_batch - self.normalize_mean) / self.normalize_std

    def apply_augmentations_gpu(self, tensor_batch):
        """Apply augmentations on GPU tensors (GPU mode only)"""
        batch_size, channels, height, width = tensor_batch.shape
        
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            tensor_batch = torch.flip(tensor_batch, dims=[3])
        
        # Random brightness/contrast adjustment
        if torch.rand(1).item() > 0.5:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.6  # ±0.3
            tensor_batch = torch.clamp(tensor_batch * brightness_factor, 0, 1)
        
        return tensor_batch

    def save_resume_checkpoint(self, category, subcategory):
        """Save the last processed category and subcategory."""
        checkpoint = {"last_category": category, "last_subcategory": subcategory}
        with open(self.resume_file, "w") as f:
            json.dump(checkpoint, f)

    def load_resume_checkpoint(self):
        """Load the last processed category and subcategory, if exists."""
        if self.resume_file.exists():
            with open(self.resume_file, "r") as f:
                return json.load(f)
        return None

    def clear_resume_checkpoint(self):
        """Remove checkpoint when everything is finished."""
        if self.resume_file.exists():
            self.resume_file.unlink()

    def is_subcategory_processed(self, split_name, category, subcategory):
        subcat_output_dir = self.output_dir / split_name / category / subcategory
        processed_file = subcat_output_dir / 'processed_data.pt'
        return processed_file.exists()

    def get_category_progress(self, category):
        processed, unprocessed = [], []
        for subcat in self.category_structure[category]:
            subcat_done = all(
                self.is_subcategory_processed(split, category, subcat)
                for split in ['train', 'val', 'test']
            )
            if subcat_done:
                processed.append(subcat)
            else:
                unprocessed.append(subcat)
        return processed, unprocessed

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

    def preprocess_video_cpu(self, video_path, augment=False, sampling_strategy='uniform'):
        """CPU-based video preprocessing (original method)"""
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

    def preprocess_video_batch_gpu(self, video_paths, augment=False, sampling_strategy='uniform'):
        """GPU-based batch video preprocessing"""
        all_processed_videos = []
        successful_filenames = []
        
        # Create progress bar for individual videos
        video_pbar = tqdm(total=len(video_paths), desc="Processing videos", unit="video")
        
        # Process videos in batches
        total_batches = (len(video_paths) + self.batch_size - 1) // self.batch_size
        batch_pbar = tqdm(total=total_batches, desc="GPU batches", unit="batch", leave=False)
        
        for i in range(0, len(video_paths), self.batch_size):
            batch_paths = video_paths[i:i+self.batch_size]
            batch_frames_list = []
            batch_filenames = []
            
            # Extract frames for all videos in batch (CPU operation)
            for video_path in batch_paths:
                try:
                    frames = self.extract_frames(video_path, self.frames_per_video, sampling_strategy)
                    if len(frames) > 0:
                        batch_frames_list.append(frames)
                        batch_filenames.append(video_path.name)
                    video_pbar.update(1)
                except Exception as e:
                    # Silently skip failed videos to keep progress smooth
                    video_pbar.update(1)
                    continue
            
            if not batch_frames_list:
                batch_pbar.update(1)
                continue
                
            # Convert to tensors and move to GPU
            batch_tensors = []
            for frames in batch_frames_list:
                frame_tensors = []
                for frame in frames:
                    pil_frame = Image.fromarray(frame)
                    # Basic resize and convert to tensor (CPU)
                    tensor_frame = self.transform_gpu(pil_frame)
                    frame_tensors.append(tensor_frame)
                
                # Pad or trim to exact frame count
                if len(frame_tensors) < self.frames_per_video:
                    padding_needed = self.frames_per_video - len(frame_tensors)
                    zero_frame = torch.zeros_like(frame_tensors[0])
                    frame_tensors.extend([zero_frame] * padding_needed)
                elif len(frame_tensors) > self.frames_per_video:
                    frame_tensors = frame_tensors[:self.frames_per_video]
                
                video_tensor = torch.stack(frame_tensors)  # Shape: [frames, channels, height, width]
                batch_tensors.append(video_tensor)
            
            if batch_tensors:
                # Stack into batch and move to GPU
                batch_tensor = torch.stack(batch_tensors).to(self.device)  # [batch, frames, channels, height, width]
                
                # Reshape for processing: [batch*frames, channels, height, width]
                original_shape = batch_tensor.shape
                reshaped_tensor = batch_tensor.view(-1, *batch_tensor.shape[2:])
                
                # Apply normalization on GPU
                normalized_tensor = self.normalize_on_gpu(reshaped_tensor)
                
                # Apply augmentations if needed
                if augment:
                    normalized_tensor = self.apply_augmentations_gpu(normalized_tensor)
                
                # Reshape back to [batch, frames, channels, height, width]
                processed_batch = normalized_tensor.view(original_shape)
                
                # Move back to CPU and add to results
                for j in range(processed_batch.shape[0]):
                    all_processed_videos.append(processed_batch[j].cpu())
                
                successful_filenames.extend(batch_filenames)
                
                # Clear GPU cache
                del batch_tensor, processed_batch, normalized_tensor, reshaped_tensor
                torch.cuda.empty_cache()
            
            batch_pbar.update(1)
        
        batch_pbar.close()
        video_pbar.close()
        
        return all_processed_videos, successful_filenames

    def process_subcategory(self, split_name, category, category_idx, subcategory, augment, sampling_strategy):
        split_input_dir = self.input_dir / split_name
        split_output_dir = self.output_dir / split_name / category / subcategory
        split_output_dir.mkdir(parents=True, exist_ok=True)

        subcat_input_dir = split_input_dir / category / subcategory
        if not subcat_input_dir.exists():
            print(f"Subcategory path {subcat_input_dir} not found in {split_name}")
            return 0

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [file for file in subcat_input_dir.iterdir()
                       if file.is_file() and file.suffix.lower() in video_extensions]

        print(f"  Processing {len(video_files)} videos in {category}/{subcategory} ({split_name}) with '{sampling_strategy}' strategy...")
        print(f"  Mode: {self.processing_mode.upper()}")

        if self.processing_mode == 'gpu':
            # GPU batch processing
            all_processed_videos, successful_filenames = self.preprocess_video_batch_gpu(
                video_files, augment=augment, sampling_strategy=sampling_strategy
            )
            
            if all_processed_videos:
                labels = [category_idx] * len(all_processed_videos)
                videos_tensor = torch.stack(all_processed_videos)
                processed_data = videos_tensor
                filenames = successful_filenames
        else:
            # CPU processing (one by one)
            processed_data = []
            labels = []
            filenames = []

            for video_path in tqdm(video_files, desc=f"{category}-{subcategory}"):
                try:
                    video_tensor = self.preprocess_video_cpu(video_path, augment=augment, sampling_strategy=sampling_strategy)
                    if video_tensor is not None:
                        processed_data.append(video_tensor)
                        labels.append(category_idx)
                        filenames.append(video_path.name)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

            if processed_data:
                processed_data = torch.stack(processed_data)

        # Save results (common for both CPU and GPU)
        if (self.processing_mode == 'gpu' and all_processed_videos) or (self.processing_mode == 'cpu' and processed_data):
            data_dict = {
                'videos': processed_data,
                'labels': torch.tensor(labels),
                'filenames': filenames,
                'category_mapping': {category: category_idx}
            }
            
            output_path = split_output_dir / 'processed_data.pt'
            torch.save(data_dict, output_path)
            print(f"  ✓ Saved {len(labels)} processed videos to {output_path}")

            metadata = {
                'num_videos': len(labels),
                'frames_per_video': self.frames_per_video,
                'image_size': self.img_size,
                'category_mapping': {category: category_idx},
                'augmented': augment,
                'selected_category': category,
                'selected_subcategory': subcategory,
                'sampling_strategy': sampling_strategy,
                'processing_mode': self.processing_mode,
                'device_used': str(self.device),
                'batch_size_used': self.batch_size if self.processing_mode == 'gpu' else 1
            }
            with open(split_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            self.save_resume_checkpoint(category, subcategory)
            return len(labels)
        else:
            print(f"  ✗ No videos processed in {category}/{subcategory} for split {split_name}.")
            return 0

    def get_preprocessing_mode(self):
        """Get preprocessing mode: interactive or bulk"""
        print("\n" + "="*50)
        print("PREPROCESSING MODE SELECTION")
        print("="*50)
        print("1. Interactive Mode - Configure sampling strategy for each subcategory one by one")
        print("2. Bulk Mode - Configure all sampling strategies first, then process everything at once")
        
        while True:
            try:
                choice = int(input("\nSelect preprocessing mode (1 or 2): ").strip())
                if choice == 1:
                    return 'interactive'
                elif choice == 2:
                    return 'bulk'
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def collect_bulk_strategies(self):
        """Collect sampling strategies for all unprocessed subcategories"""
        print("\n" + "="*40)
        print("BULK STRATEGY CONFIGURATION")
        print("="*40)
        
        strategies = {}
        
        for category in self.all_categories:
            processed, unprocessed = self.get_category_progress(category)
            if len(unprocessed) > 0:
                print(f"\nCategory: {category}")
                print(f"Unprocessed subcategories: {len(unprocessed)}")
                
                strategies[category] = {}
                for subcat in unprocessed:
                    strategy = self.get_sampling_strategy_for_subcategory(category, subcat)
                    strategies[category][subcat] = strategy
        
        return strategies

    def process_bulk_mode(self):
        """Process all categories using pre-configured strategies"""
        # Collect all strategies first
        strategies = self.collect_bulk_strategies()
        
        if not strategies:
            print("No unprocessed subcategories found.")
            return
        
        print("\n" + "="*50)
        print("STARTING BULK PROCESSING")
        print("="*50)
        
        # Show summary of what will be processed
        total_subcats = sum(len(subcats) for subcats in strategies.values())
        print(f"Total subcategories to process: {total_subcats}")
        
        for category, subcats in strategies.items():
            print(f"\n{category}: {list(subcats.keys())}")
            for subcat, strategy in subcats.items():
                print(f"  - {subcat}: {strategy}")
        
        input("\nPress Enter to start bulk processing...")
        
        resume_point = self.load_resume_checkpoint()
        start_from_category = None
        start_from_subcat = None
        if resume_point:
            start_from_category = resume_point["last_category"]
            start_from_subcat = resume_point["last_subcategory"]
            print(f"\nResuming from last checkpoint: Category '{start_from_category}', Subcategory '{start_from_subcat}'")
        
        # Process everything
        for category, subcats in strategies.items():
            category_idx = self.all_categories.index(category)
            
            # If resuming, skip categories/subcategories until we reach the checkpoint
            resume_mode = (category == start_from_category) if start_from_category else False
            skip = bool(resume_point)
            
            print(f"\n" + "="*30)
            print(f"Processing Category: {category}")
            print("="*30)
            
            for subcat, strategy in subcats.items():
                if skip and resume_mode:
                    if subcat == start_from_subcat:  # Skip the last processed subcategory
                        print(f"Skipping already processed subcategory '{subcat}' due to resume checkpoint...")
                        continue
                    else:
                        skip = False  # resume from here
                
                print(f"\nProcessing subcategory: {subcat} (strategy: {strategy})")
                
                # Process all splits for this subcategory
                for split in ['train', 'val', 'test']:
                    augment = (split == 'train')
                    print(f"\n--- Processing {split} split ---")
                    self.process_subcategory(split, category, category_idx, subcat, augment, strategy)
            
            print(f"Completed category: {category}")
        
        print("\n" + "="*50)
        print("BULK PROCESSING COMPLETED!")
        print("="*50)

    def process_interactive_mode(self):
        """Process categories interactively (original method)"""
        resume_point = self.load_resume_checkpoint()
        start_from_category = None
        start_from_subcat = None
        if resume_point:
            start_from_category = resume_point["last_category"]
            start_from_subcat = resume_point["last_subcategory"]
            print(f"\nResuming from last checkpoint: Category '{start_from_category}', Subcategory '{start_from_subcat}'")

        while True:
            print("\nAvailable main categories:")
            for i, cat in enumerate(self.all_categories, 1):
                processed, unprocessed = self.get_category_progress(cat)
                if len(unprocessed) == 0:
                    status = "(fully processed)"
                elif len(processed) == 0:
                    status = "(not processed)"
                else:
                    status = f"(partially processed: Done {len(processed)}, Remaining {len(unprocessed)})"
                print(f"  {i}. {cat} {status}")
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
            processed, unprocessed = self.get_category_progress(selected_category)

            if len(unprocessed) == 0:
                print(f"Category '{selected_category}' is already fully processed. Skipping...")
                continue

            category_idx = self.all_categories.index(selected_category)

            # If resuming, skip subcategories until we reach last one
            resume_mode = (selected_category == start_from_category) if start_from_category else False
            skip = bool(resume_point)

            for subcat in unprocessed:
                if skip and resume_mode:
                    if subcat == start_from_subcat:  # Skip the last processed subcategory
                        print(f"Skipping already processed subcategory '{subcat}' due to resume checkpoint...")
                        continue
                    else:
                        skip = False  # resume from here

                sampling_strategy = self.get_sampling_strategy_for_subcategory(selected_category, subcat)
                
                # Process all splits for this subcategory
                for split in ['train', 'val', 'test']:
                    augment = (split == 'train')
                    print(f"\n--- Processing {split} split ---")
                    self.process_subcategory(split, selected_category, category_idx, subcat, augment, sampling_strategy)

            print(f"Completed processing for remaining subcategories in '{selected_category}'.")

    def interactive_process(self):
        # Setup processing mode first
        self.setup_processing_mode()
        
        # Get preprocessing mode
        preprocessing_mode = self.get_preprocessing_mode()
        
        if preprocessing_mode == 'interactive':
            self.process_interactive_mode()
        else:  # bulk mode
            self.process_bulk_mode()
        
        self.clear_resume_checkpoint()
        print("\nAll selected processing done. Goodbye!")


def main():
    preprocessor = VideoPreprocessor(
        input_dir="video_classification_project/data/raw",
        output_dir="video_classification_project/data/processed",
        frames_per_video=32,
        img_size=(224, 224),
        clip_duration=10
    )
    preprocessor.interactive_process()


if __name__ == "__main__":
    main()