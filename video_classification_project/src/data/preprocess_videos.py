# CRITICAL: Set CUDA device BEFORE importing torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import random


class VideoPreprocessor:
    def __init__(self, input_dir, output_dir, frames_per_video=64, img_size=(256, 256), clip_duration=10):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.clip_duration = clip_duration
        
        # Initialize processing mode
        self.processing_mode = None
        self.device = None
        self.batch_size = 1  # Default for CPU
        
        # Enhanced CPU transforms with aspect ratio preservation
        self.transform = transforms.Compose([
            transforms.Resize(int(img_size[0] * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # GPU-specific attributes
        self.normalize_mean = None
        self.normalize_std = None
        
        self.category_structure = {
            'Animation': ['Cartoon', 'Animation', 'Lego minifigure', 'Naruto',
                          'The Walt Disney Company', 'Dragon Ball', 'Sonic the Hedgehog',
                          'One Piece', 'Bleach'],
            'Gaming': ['Games', 'Video game', 'Minecraft', 'Call of Duty', 'Grand Theft Auto', 'Grand Theft Auto V',
                       'World of Warcraft', 'League of Legends', 'Battlefield', 'RuneScape',
                       'Action-adventure game', 'FIFA 15'],
            'Natural_Content': ['Animal', 'Pet', 'Fishing', 'Fish', 'Outdoor recreation', 'Dog',
                                'Horse', 'Bird', 'Plant', 'Cat', 'Farm', 'Garden', 'Nature',
                                'Tree', 'Wildlife', 'Chicken'],
            'Flat_Content': ['Website', 'Chart', 'Map', 'Logo', 'Text', 'Typography',
                             'Screencast', 'Illustration', 'Poster']
        }
        self.all_categories = list(self.category_structure.keys())
        self.resume_file = self.output_dir / "resume_checkpoint.json"

    def setup_processing_mode(self):
        """Setup CPU or GPU processing mode based on user choice - MIG COMPATIBLE"""
        print("\n" + "="*50)
        print("VIDEO PREPROCESSOR - PROCESSING MODE SELECTION")
        print("="*50)
        
        print(f"\nProcessing Mode Options:")
        print(f"1. CPU Mode - Compatible with all systems (slower)")
        print(f"2. GPU Mode - Faster processing (A100 MIG detected)")
        
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
                    # CRITICAL FIX: Check CUDA availability WITHOUT triggering initialization
                    try:
                        # Set environment variable BEFORE any CUDA operations
                        import os
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        
                        # Now check availability (should work with MIG)
                        if not torch.cuda.is_available():
                            print("GPU not available. Please select CPU mode (1).")
                            continue
                        
                        self.processing_mode = 'gpu'
                        
                        # Use cuda:0 explicitly (MIG-safe)
                        self.device = torch.device('cuda:0')
                        
                        # Set as current device
                        torch.cuda.set_device(0)
                        
                        # Clear any cached state
                        torch.cuda.empty_cache()
                        
                        self._setup_gpu_components()
                        print(f"Selected: GPU Mode")
                        
                        # Get device name safely
                        try:
                            gpu_name = torch.cuda.get_device_name(0)
                            print(f"GPU: {gpu_name}")
                        except:
                            print(f"GPU: CUDA Device 0 (MIG Instance)")
                        
                        break
                        
                    except Exception as e:
                        print(f"Error initializing GPU: {e}")
                        print("Falling back to CPU mode (1).")
                        self.processing_mode = 'cpu'
                        self.device = torch.device('cpu')
                        self.batch_size = 1
                        break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        print(f"Using device: {self.device}")
        print("="*50 + "\n")

    def _setup_gpu_components(self):
        """Setup GPU-specific components - MIG OPTIMIZED WITH SAFE INITIALIZATION"""
        # Basic transforms (no device operations yet)
        self.transform_gpu = transforms.Compose([
            transforms.Resize(int(self.img_size[0] * 1.15)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])
        
        # CRITICAL FIX: Create tensors directly on the device
        # Use index 0 explicitly for MIG compatibility
        try:
            self.normalize_mean = torch.tensor(
                [0.485, 0.456, 0.406], 
                device='cuda:0',  # Explicit device string
                dtype=torch.float32
            ).view(3, 1, 1)
            
            self.normalize_std = torch.tensor(
                [0.229, 0.224, 0.225], 
                device='cuda:0',  # Explicit device string
                dtype=torch.float32
            ).view(3, 1, 1)
        except Exception as e:
            print(f"Error creating normalization tensors: {e}")
            raise
        
        # Test GPU operation to ensure initialization worked
        try:
            test_tensor = torch.randn(1, 3, 256, 256, device='cuda:0')
            test_norm = (test_tensor - self.normalize_mean) / self.normalize_std
            del test_tensor, test_norm
            torch.cuda.empty_cache()
            print("✓ GPU initialization test passed")
        except Exception as e:
            print(f"⚠ GPU initialization test failed: {e}")
            raise RuntimeError("GPU initialization failed - try CPU mode instead")
        
        # CRITICAL: Detect actual available memory (MIG-aware)
        try:
            # Get free memory using device index
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            
            print(f"GPU Memory: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total")
            
            # Calculate batch size based on ACTUAL free memory
            # Each video tensor: 64 frames × 3 channels × 256 × 256 × 4 bytes (float32)
            bytes_per_video = 64 * 3 * 256 * 256 * 4
            safe_memory = free_memory * 0.4  # Use only 40% for MIG safety
            max_batch = int(safe_memory / bytes_per_video)
            
            # Very conservative batch size for MIG
            self.batch_size = min(4, max(1, max_batch))
            
            print(f"Optimized batch size for MIG: {self.batch_size}")
            print(f"Estimated memory per batch: {(bytes_per_video * self.batch_size) / 1e9:.2f} GB")
            
        except Exception as e:
            print(f"Warning: Could not detect GPU memory: {e}")
            print("Using conservative batch size: 1")
            self.batch_size = 1

    def normalize_on_gpu(self, tensor_batch):
        """Perform normalization on GPU for better performance"""
        return (tensor_batch - self.normalize_mean) / self.normalize_std

    def apply_augmentations_gpu(self, tensor_batch, category=None):
        """Apply augmentations on GPU tensors with category-specific enhancements"""
        batch_size, channels, height, width = tensor_batch.shape
        
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            tensor_batch = torch.flip(tensor_batch, dims=[3])
        
        # Category-specific augmentations
        if category in ['Gaming', 'Animation']:
            # Motion blur simulation
            if torch.rand(1).item() > 0.6:
                blur_kernel = 3
                blurred = torch.nn.functional.avg_pool2d(tensor_batch, kernel_size=blur_kernel, stride=1, padding=1)
                alpha = 0.3 + torch.rand(1).item() * 0.4
                tensor_batch = alpha * blurred + (1 - alpha) * tensor_batch
            
            # Stronger color adjustments
            if torch.rand(1).item() > 0.5:
                brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.8
                contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
                tensor_batch = torch.clamp(tensor_batch * brightness_factor, 0, 1)
                tensor_batch = torch.clamp((tensor_batch - 0.5) * contrast_factor + 0.5, 0, 1)
        else:
            # Moderate augmentations
            if torch.rand(1).item() > 0.5:
                brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                tensor_batch = torch.clamp(tensor_batch * brightness_factor, 0, 1)
        
        return tensor_batch

    def save_resume_checkpoint(self, category, subcategory):
        checkpoint = {"last_category": category, "last_subcategory": subcategory}
        with open(self.resume_file, "w") as f:
            json.dump(checkpoint, f)

    def load_resume_checkpoint(self):
        if self.resume_file.exists():
            with open(self.resume_file, "r") as f:
                return json.load(f)
        return None

    def clear_resume_checkpoint(self):
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
        print("1. Uniform - Sample frames across entire video")
        print("2. Middle Clip - Focus on middle 10 seconds")
        print("3. Random Clip - Random 10-second clips")
        print("4. Multi-Clip - Sample from multiple segments (early, middle, late)")
        print("5. Adaptive Multi-Clip - Smart multi-clip for Gaming/Animation, middle for others")
        
        while True:
            strategy_input = input(f"Sampling strategy (1-5) for {subcategory}: ").strip()
            if strategy_input in ['1', '2', '3', '4', '5']:
                strategies = {
                    '1': 'uniform',
                    '2': 'middle_clip',
                    '3': 'random_clip',
                    '4': 'multi_clip',
                    '5': 'adaptive_multi_clip'
                }
                return strategies[strategy_input]
            else:
                print("Invalid input. Please enter 1, 2, 3, 4, or 5.")

    def extract_frames_multi_clip(self, video_path, max_frames=64, num_clips=3):
        """Extract frames from multiple segments of the video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        if total_frames == 0 or duration < 1.0:
            print(f"  WARNING: Invalid video (frames={total_frames}, duration={duration:.2f}s): {video_path.name}")
            cap.release()
            return []
        
        frames = []
        frames_per_clip = max_frames // num_clips
        clip_positions = [0.2, 0.5, 0.8]
        
        for position in clip_positions[:num_clips]:
            clip_center = int(total_frames * position)
            clip_duration_frames = min(int(self.clip_duration * fps), total_frames // num_clips)
            start_frame = max(0, clip_center - clip_duration_frames // 2)
            end_frame = min(total_frames, start_frame + clip_duration_frames)
            
            if end_frame - start_frame <= frames_per_clip:
                frame_indices = list(range(start_frame, end_frame))
            else:
                step = (end_frame - start_frame) / frames_per_clip
                frame_indices = [start_frame + int(i * step) for i in range(frames_per_clip)]
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
        
        cap.release()
        return frames[:max_frames]

    def extract_frames(self, video_path, max_frames=64, sampling_strategy='uniform', category=None):
        """Enhanced frame extraction with multiple strategies"""
        if sampling_strategy == 'adaptive_multi_clip':
            if category in ['Gaming', 'Animation']:
                return self.extract_frames_multi_clip(video_path, max_frames, num_clips=3)
            else:
                sampling_strategy = 'middle_clip'
        
        if sampling_strategy == 'multi_clip':
            return self.extract_frames_multi_clip(video_path, max_frames, num_clips=3)
        
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

    def get_category_specific_augmentation(self, category, augment):
        """Create category-specific augmentation transforms"""
        if not augment:
            return transforms.Compose([
                transforms.Resize(int(self.img_size[0] * 1.15)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        base_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(int(self.img_size[0] * 1.15)),
        ]
        
        if category in ['Gaming', 'Animation']:
            category_transforms = [
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
                transforms.RandomResizedCrop(self.img_size[0], scale=(0.75, 1.0)),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
            ]
        elif category == 'Natural_Content':
            category_transforms = [
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomCrop(self.img_size[0], padding=4),
            ]
        else:
            category_transforms = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.CenterCrop(self.img_size),
            ]
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(base_transforms + category_transforms + final_transforms)

    def preprocess_video_cpu(self, video_path, augment=False, sampling_strategy='uniform', category=None):
        """CPU-based video preprocessing"""
        frames = self.extract_frames(video_path, self.frames_per_video, sampling_strategy, category)
        if len(frames) == 0:
            return None

        augment_transform = self.get_category_specific_augmentation(category, augment)
        
        processed_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            if augment:
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

    def preprocess_video_batch_gpu(self, video_paths, augment=False, sampling_strategy='uniform', category=None):
        """GPU-based batch video preprocessing - MIG OPTIMIZED"""
        all_processed_videos = []
        successful_filenames = []
        
        video_pbar = tqdm(total=len(video_paths), desc="Processing videos", unit="video")
        total_batches = (len(video_paths) + self.batch_size - 1) // self.batch_size
        batch_pbar = tqdm(total=total_batches, desc="GPU batches", unit="batch", leave=False)
        
        for i in range(0, len(video_paths), self.batch_size):
            batch_paths = video_paths[i:i+self.batch_size]
            batch_frames_list = []
            batch_filenames = []
            
            # Extract frames (CPU operation)
            for video_path in batch_paths:
                try:
                    frames = self.extract_frames(video_path, self.frames_per_video, sampling_strategy, category)
                    if len(frames) > 0:
                        batch_frames_list.append(frames)
                        batch_filenames.append(video_path.name)
                    video_pbar.update(1)
                except Exception:
                    video_pbar.update(1)
                    continue
            
            if not batch_frames_list:
                batch_pbar.update(1)
                continue
                
            # Convert to tensors
            batch_tensors = []
            for frames in batch_frames_list:
                frame_tensors = []
                for frame in frames:
                    pil_frame = Image.fromarray(frame)
                    tensor_frame = self.transform_gpu(pil_frame)
                    frame_tensors.append(tensor_frame)
                
                if len(frame_tensors) < self.frames_per_video:
                    padding_needed = self.frames_per_video - len(frame_tensors)
                    zero_frame = torch.zeros_like(frame_tensors[0])
                    frame_tensors.extend([zero_frame] * padding_needed)
                elif len(frame_tensors) > self.frames_per_video:
                    frame_tensors = frame_tensors[:self.frames_per_video]
                
                video_tensor = torch.stack(frame_tensors)
                batch_tensors.append(video_tensor)
            
            if batch_tensors:
                # Move to GPU and process
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                original_shape = batch_tensor.shape
                reshaped_tensor = batch_tensor.view(-1, *batch_tensor.shape[2:])
                
                normalized_tensor = self.normalize_on_gpu(reshaped_tensor)
                
                if augment:
                    normalized_tensor = self.apply_augmentations_gpu(normalized_tensor, category)
                
                processed_batch = normalized_tensor.view(original_shape)
                
                # Move back to CPU
                for j in range(processed_batch.shape[0]):
                    all_processed_videos.append(processed_batch[j].cpu())
                
                successful_filenames.extend(batch_filenames)
                
                # CRITICAL: Clear GPU cache aggressively for MIG
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
        print(f"  Frames per video: {self.frames_per_video}, Resolution: {self.img_size}")

        if self.processing_mode == 'gpu':
            all_processed_videos, successful_filenames = self.preprocess_video_batch_gpu(
                video_files, augment=augment, sampling_strategy=sampling_strategy, category=category
            )
            
            if all_processed_videos:
                labels = [category_idx] * len(all_processed_videos)
                videos_tensor = torch.stack(all_processed_videos)
                processed_data = videos_tensor
                filenames = successful_filenames
        else:
            processed_data = []
            labels = []
            filenames = []

            for video_path in tqdm(video_files, desc=f"{category}-{subcategory}"):
                try:
                    video_tensor = self.preprocess_video_cpu(
                        video_path, augment=augment, 
                        sampling_strategy=sampling_strategy, 
                        category=category
                    )
                    if video_tensor is not None:
                        processed_data.append(video_tensor)
                        labels.append(category_idx)
                        filenames.append(video_path.name)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

            if processed_data:
                processed_data = torch.stack(processed_data)

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
        print("\n" + "="*50)
        print("PREPROCESSING MODE SELECTION")
        print("="*50)
        print("1. Interactive Mode - Configure sampling strategy for each subcategory")
        print("2. Bulk Mode - Configure all strategies first, then process")
        
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
        print("\n" + "="*40)
        print("BULK STRATEGY CONFIGURATION")
        print("="*40)
        
        strategies = {}
        
        for category in self.all_categories:
            processed, unprocessed = self.get_category_progress(category)
            if len(unprocessed) > 0:
                print(f"\nCategory: {category}")
                print(f"Unprocessed subcategories: {len(unprocessed)}")
                
                if category in ['Gaming', 'Animation']:
                    print(f"Recommendation: Use 'Adaptive Multi-Clip' for {category}")
                
                strategies[category] = {}
                for subcat in unprocessed:
                    strategy = self.get_sampling_strategy_for_subcategory(category, subcat)
                    strategies[category][subcat] = strategy
        
        return strategies

    def process_bulk_mode(self):
        strategies = self.collect_bulk_strategies()
        
        if not strategies:
            print("No unprocessed subcategories found.")
            return
        
        print("\n" + "="*50)
        print("STARTING BULK PROCESSING")
        print("="*50)
        
        total_subcats = sum(len(subcats) for subcats in strategies.values())
        print(f"Total subcategories to process: {total_subcats}")
        print(f"Frames per video: {self.frames_per_video}")
        print(f"Resolution: {self.img_size}")
        
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
            print(f"\nResuming from: '{start_from_category}' - '{start_from_subcat}'")
        
        for category, subcats in strategies.items():
            category_idx = self.all_categories.index(category)
            resume_mode = (category == start_from_category) if start_from_category else False
            skip = bool(resume_point)
            
            print(f"\n{'='*30}\nProcessing Category: {category}\n{'='*30}")
            
            for subcat, strategy in subcats.items():
                if skip and resume_mode:
                    if subcat == start_from_subcat:
                        print(f"Skipping processed '{subcat}'...")
                        continue
                    else:
                        skip = False
                
                print(f"\nProcessing: {subcat} (strategy: {strategy})")
                
                for split in ['train', 'val', 'test']:
                    augment = (split == 'train')
                    print(f"\n--- Processing {split} split ---")
                    self.process_subcategory(split, category, category_idx, subcat, augment, strategy)
            
            print(f"Completed category: {category}")
        
        print("\n" + "="*50)
        print("BULK PROCESSING COMPLETED!")
        print("="*50)

    def process_interactive_mode(self):
        resume_point = self.load_resume_checkpoint()
        start_from_category = None
        start_from_subcat = None
        if resume_point:
            start_from_category = resume_point["last_category"]
            start_from_subcat = resume_point["last_subcategory"]
            print(f"\nResuming from: '{start_from_category}' - '{start_from_subcat}'")

        while True:
            print("\nAvailable main categories:")
            for i, cat in enumerate(self.all_categories, 1):
                processed, unprocessed = self.get_category_progress(cat)
                if len(unprocessed) == 0:
                    status = "(fully processed)"
                elif len(processed) == 0:
                    status = "(not processed)"
                else:
                    status = f"(partial: {len(processed)} done, {len(unprocessed)} remaining)"
                print(f"  {i}. {cat} {status}")
            print("  0. Exit")

            try:
                choice = int(input("\nSelect category (or 0 to exit): ").strip())
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
                print(f"'{selected_category}' already fully processed.")
                continue

            category_idx = self.all_categories.index(selected_category)

            if selected_category in ['Gaming', 'Animation']:
                print(f"\n💡 Recommendation: Use 'Adaptive Multi-Clip' for {selected_category}")
                print("   Captures multiple temporal segments for dynamic content.")

            resume_mode = (selected_category == start_from_category) if start_from_category else False
            skip = bool(resume_point)

            for subcat in unprocessed:
                if skip and resume_mode:
                    if subcat == start_from_subcat:
                        print(f"Skipping processed '{subcat}'...")
                        continue
                    else:
                        skip = False

                sampling_strategy = self.get_sampling_strategy_for_subcategory(selected_category, subcat)
                
                for split in ['train', 'val', 'test']:
                    augment = (split == 'train')
                    print(f"\n--- Processing {split} split ---")
                    self.process_subcategory(split, selected_category, category_idx, subcat, augment, sampling_strategy)

            print(f"Completed '{selected_category}'.")

    def interactive_process(self):
        print("\n" + "="*60)
        print("ENHANCED VIDEO PREPROCESSOR v2.0 - SERVER OPTIMIZED")
        print("="*60)
        print("\nKey Features:")
        print("✓ 64 frames per video (2× temporal coverage)")
        print("✓ 256×256 resolution (higher detail)")
        print("✓ Multi-clip sampling for Gaming/Animation")
        print("✓ Category-specific augmentations")
        print("✓ MIG-aware GPU batch processing")
        print("="*60)
        
        self.setup_processing_mode()
        preprocessing_mode = self.get_preprocessing_mode()
        
        if preprocessing_mode == 'interactive':
            self.process_interactive_mode()
        else:
            self.process_bulk_mode()
        
        self.clear_resume_checkpoint()
        print("\nAll processing completed. Goodbye!")


def main():
    preprocessor = VideoPreprocessor(
        input_dir="/workspace/NVIDIA-Video-Classification-Project/video_classification_project/data/raw",
        output_dir="/workspace/NVIDIA-Video-Classification-Project/video_classification_project/data/processed",
        frames_per_video=64,
        img_size=(256, 256),
        clip_duration=10
    )
    preprocessor.interactive_process()


if __name__ == "__main__":
    main()