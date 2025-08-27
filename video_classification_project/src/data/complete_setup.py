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
    
    # Check for category directories
    required_categories = ["Animation", "Gaming", "Natural Content", "Flat Content"]
    missing_categories = []
    
    for category in required_categories:
        category_path = dataset_dir / category / "videos"
        if not category_path.exists():
            missing_categories.append(category)
    
    if missing_categories:
        print(f"❌ ERROR: Missing category directories: {missing_categories}")
        print("Please ensure each category has a 'videos' subfolder.")
        return False
    
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
    
    from analyze_dataset import analyze_dataset  # Import your analyzer
    
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
        import analyze_dataset
        category_counts, total_videos = analyze_dataset.analyze_dataset("Dataset")
        
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
        import split_dataset
        split_dataset.main()
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
        import preprocess_videos
        
        # For quick test, reduce frames per video
        frames_per_video = 10 if quick_test else 30
        
        preprocessor = preprocess_videos.VideoPreprocessor(
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