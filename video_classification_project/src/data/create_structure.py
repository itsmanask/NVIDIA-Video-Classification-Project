import os
import shutil
from pathlib import Path

def create_project_structure(base_path="video_classification_project"):
    """
    Create the recommended folder structure for video classification project
    """
    
    # Define the folder structure
    folders = [
        # Data folders
        "data/raw/train/Animation/Cartoon",
        "data/raw/train/Animation/Animation", 
        "data/raw/train/Animation/Pokemon",
        "data/raw/train/Gaming/Games",
        "data/raw/train/Gaming/Video_game",
        "data/raw/train/Natural_Content/Animal",
        "data/raw/train/Natural_Content/Pet",
        "data/raw/train/Flat_Content/Art",
        "data/raw/train/Flat_Content/Drawing",
        
        "data/raw/val/Animation/Cartoon",
        "data/raw/val/Animation/Animation",
        "data/raw/val/Animation/Pokemon", 
        "data/raw/val/Gaming/Games",
        "data/raw/val/Gaming/Video_game",
        "data/raw/val/Natural_Content/Animal",
        "data/raw/val/Natural_Content/Pet",
        "data/raw/val/Flat_Content/Art",
        "data/raw/val/Flat_Content/Drawing",
        
        "data/raw/test/Animation/Cartoon",
        "data/raw/test/Animation/Animation",
        "data/raw/test/Animation/Pokemon",
        "data/raw/test/Gaming/Games", 
        "data/raw/test/Gaming/Video_game",
        "data/raw/test/Natural_Content/Animal",
        "data/raw/test/Natural_Content/Pet",
        "data/raw/test/Flat_Content/Art",
        "data/raw/test/Flat_Content/Drawing",
        
        # Processed data folders
        "data/processed/train",
        "data/processed/val", 
        "data/processed/test",
        
        # Other project folders
        "models",
        "notebooks", 
        "src/data",
        "src/models",
        "src/utils",
        "configs",
        "results"
    ]
    
    # Create base directory
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True)
    
    # Create all folders
    for folder in folders:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder_path}")
    
    # Create requirements.txt
    requirements = """torch>=1.9.0
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
    
    with open(base_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    
    print(f"\nProject structure created successfully at: {base_dir.absolute()}")
    print("Next: Install requirements with: pip install -r requirements.txt")

if _name_ == "_main_":
    create_project_structure()