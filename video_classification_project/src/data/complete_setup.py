#!/usr/bin/env python3
"""
Complete Video Classification Project Setup
Run this script from the PR1 directory to set up everything automatically.

Expected directory structure before running:
PR1/
├── Dataset/
│   ├── Animation/videos/
│   ├── Gaming/videos/
│   ├── Natural Content/videos/
│   └── Flat Content/videos/
└── [this script]

This script will create:
PR1/
├── Dataset/ (existing)
├── video_classification_project/
│   ├── data/
│   │   ├── raw/ (organized train/val/test splits)
│   │   └── processed/ (preprocessed tensors)
│   ├── models/
│   ├── src/
│   └── requirements.txt
└── [analysis and setup scripts]
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if we're in the right directory and have the required structure"""
    print("🔍 Checking environment...")
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if Dataset directory exists
    dataset_dir = Path("Dataset")
    if not dataset_dir.exists():
        print("❌ ERROR: 'Dataset' directory not found!")
        print("Please ensure you're running this script from the PR1 directory")
        print("and that the Dataset folder exists.")
        return False
    
    # Check for main category directories
    required_categories = ["Animation", "Gaming", "Natural Content", "Flat Content"]
    missing_categories = []
    
    for category in required_categories:
        category_path = dataset_dir / category / "videos"
        if not category_path.exists():
            missing_categories.append(category)
    
    if missing_categories:
        print(f"❌ ERROR: Missing category directories: {missing_categories}")
        print("Please ensure each category has the structure: Dataset/Category/videos/Subcategory/")
        return False
    
    # Check for some subcategories to verify structure
    sample_subcategories = {
        'Animation': ['Cartoon', 'Animation', 'Lego minifigure'],
        'Gaming': ['Games', 'Video game'],
        'Natural Content': ['Animal', 'Pet'],
        'Flat Content': ['Art', 'Drawing']
    }
    
    # Use case-insensitive extension matching to avoid duplicates
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    found_videos = False
    for main_cat, subcats in sample_subcategories.items():
        for subcat in subcats[:2]:  # Check first 2 subcategories
            subcat_path = dataset_dir / main_cat / "videos" / subcat
            if subcat_path.exists():
                # Check if it has any video files - fixed to avoid double counting
                all_files = list(subcat_path.iterdir())
                video_files = [
                    f for f in all_files 
                    if f.is_file() and f.suffix.lower() in video_extensions
                ]
                
                if video_files:  # If any videos found
                    found_videos = True
                    break
        if found_videos:
            break
    
    if not found_videos:
        print("⚠️  WARNING: No video files found in expected subcategories")
        print("Expected structure: Dataset/Animation/videos/Cartoon/*.mp4")
    else:
        print("✅ Environment check passed!")
    
    return True

def install_requirements():
    """Install required Python packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.61.0",
        "Pillow>=8.3.0"
    ]
    
    try:
        import subprocess
        for req in requirements:
            print(f"Installing {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", req.split('>=')[0]], 
                          check=True, capture_output=True)
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        print("Please install manually: pip install torch torchvision opencv-python numpy pandas matplotlib tqdm Pillow")
        return False

def create_project_structure():
    """Create the project directory structure"""
    print("\n📁 Creating project structure...")
    
    folders = [
        "video_classification_project/data/raw/train/Animation",
        "video_classification_project/data/raw/train/Gaming", 
        "video_classification_project/data/raw/train/Natural_Content",
        "video_classification_project/data/raw/train/Flat_Content",
        
        "video_classification_project/data/raw/val/Animation",
        "video_classification_project/data/raw/val/Gaming",
        "video_classification_project/data/raw/val/Natural_Content", 
        "video_classification_project/data/raw/val/Flat_Content",
        
        "video_classification_project/data/raw/test/Animation",
        "video_classification_project/data/raw/test/Gaming",
        "video_classification_project/data/raw/test/Natural_Content",
        "video_classification_project/data/raw/test/Flat_Content",
        
        "video_classification_project/data/processed/train",
        "video_classification_project/data/processed/val",
        "video_classification_project/data/processed/test",
        
        "video_classification_project/models",
        "video_classification_project/notebooks",
        "video_classification_project/src/data",
        "video_classification_project/src/models", 
        "video_classification_project/src/utils",
        "video_classification_project/configs",
        "video_classification_project/results"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Create requirements.txt
    requirements_content = """torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.61.0
tensorboard>=2.6.0
Pillow>=8.3.0
jupyter>=1.0.0
"""
    
    with open("video_classification_project/requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Project structure created!")

def run_dataset_analysis():
    """Analyze the current dataset"""
    print("\n📊 Analyzing dataset...")
    
    try:
        # Import and run the analyzer
        sys.path.append('.')  # Add current directory to path
        
        # Define the analyzer function inline if import fails
        try:
            from analyze_dataset import analyze_dataset
            category_counts, total_videos = analyze_dataset("Dataset")
        except ImportError:
            print("Analyzer import failed, running basic count...")
            # Basic count without importing - fixed to avoid double counting
            dataset_path = Path("Dataset")
            total_videos = 0
            
            categories = ["Animation", "Gaming", "Natural Content", "Flat Content"]
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}  # Use set for case-insensitive matching
            
            for category in categories:
                category_path = dataset_path / category / "videos"
                if category_path.exists():
                    # Use rglob to recursively find all files, then filter by extension
                    all_files = list(category_path.rglob("*"))
                    video_files = [
                        f for f in all_files 
                        if f.is_file() and f.suffix.lower() in video_extensions
                    ]
                    total_videos += len(video_files)
        
        if total_videos == 0:
            print("❌ No videos found in dataset!")
            return False
            
        print(f"✅ Found {total_videos} total videos")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def run_video_splitting():
    """Split videos into train/val/test sets"""
    print("\n✂️ Splitting dataset...")
    
    try:
        # Import and run the splitter
        sys.path.append('.')
        from split_dataset import VideoDatasetSplitter
        
        splitter = VideoDatasetSplitter(
            source_base_dir="Dataset",
            target_dir="video_classification_project/data",
            train_ratio=0.7,
            val_ratio=0.2, 
            test_ratio=0.1
        )
        
        splitter.split_videos_by_category()
        splitter.verify_split()
        print("✅ Dataset split completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during splitting: {e}")
        return False

def run_preprocessing(quick_test=False):
    """Preprocess videos into tensors"""
    print("\n🔄 Preprocessing videos...")
    
    if quick_test:
        print("Running quick test with reduced parameters...")
    
    try:
        # Import and run the preprocessor
        sys.path.append('.')
        from preprocess_videos import VideoPreprocessor
        
        # For quick test, reduce frames per video
        frames_per_video = 10 if quick_test else 30
        
        preprocessor = VideoPreprocessor(
            input_dir="video_classification_project/data/raw",
            output_dir="video_classification_project/data/processed",
            frames_per_video=frames_per_video,
            img_size=(224, 224)
        )
        
        preprocessor.process_all_splits()
        print("✅ Preprocessing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Video Classification Project Setup")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        return
    
    # Step 2: Create project structure
    create_project_structure()
    
    # Step 3: Analyze dataset
    if not run_dataset_analysis():
        print("⚠️ Dataset analysis failed, but continuing...")
    
    # Ask user if they want to proceed with full setup
    response = input("\n❓ Do you want to proceed with dataset splitting and preprocessing? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        # Step 4: Split dataset
        if run_video_splitting():
            
            # Ask about preprocessing
            preprocess_response = input("\n❓ Preprocessing can take a while. Start now? (y/n/quick): ")
            
            if preprocess_response.lower() in ['y', 'yes']:
                run_preprocessing(quick_test=False)
            elif preprocess_response.lower() == 'quick':
                run_preprocessing(quick_test=True)
            else:
                print("⏸️ Skipping preprocessing. You can run it later.")
        
        print("\n🎉 Setup completed!")
        print("\nNext steps:")
        print("1. cd video_classification_project")
        print("2. Start building your CNN-LSTM model")
        print("3. Use the preprocessed data in data/processed/")
        
    else:
        print("⏸️ Setup paused. Project structure created.")
        print("You can run individual scripts manually when ready.")

if __name__ == "__main__":
    main()