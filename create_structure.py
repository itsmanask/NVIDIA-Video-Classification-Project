import os
from pathlib import Path

def create_project_structure(base_path="video_classification_project"):
    """
    Create the full folder structure for the video classification project
    based on the folder structure from folder_structureee.txt
    """

    # Dataset categories and subcategories for RAW data
    categories = {
        "Animation": [
            "Animation", "Bleach", "Cartoon", "Dragon Ball",
            "Lego minifigure", "Naruto", "One Piece", 
            "Sonic the Hedgehog", "The Walt Disney Company"
        ],
        "Flat_Content": [
            "Chart", "Illustration", "Logo", "Map",
            "Poster", "Screencast", "Text",
            "Typography", "Website"
        ],
        "Gaming": [
            "Action-adventure game", "Battlefield", "Call of Duty",
            "FIFA 15", "Games", "Grand Theft Auto", 
            "Grand Theft Auto V", "League of Legends", 
            "Minecraft", "RuneScape", "Video game", 
            "World of Warcraft"
        ],
        "Natural_Content": [
            "Animal", "Bird", "Cat", "Chicken", "Dog",
            "Farm", "Fish", "Fishing", "Garden", "Horse",
            "Nature", "Outdoor recreation", "Pet",
            "Plant", "Tree", "Wildlife"
        ]
    }

    # Base dataset splits
    splits = ["train", "val", "test"]

    # Define folders
    folders = []

    # Raw data with full subcategories
    for split in splits:
        for category, subcats in categories.items():
            for sub in subcats:
                folders.append(f"data/raw/{split}/{category}/{sub}")

    # Processed data - only main categories (no subcategories)
    for split in splits:
        for category in categories.keys():
            folders.append(f"data/processed/{split}/{category}")

    # Special case: train/Animation/Cartoon subfolder in processed
    folders.append("data/processed/train/Animation/Cartoon")

    # Project support folders
    folders += [
        "models_enhanced",
        "features_enhanced",
        "results",
        "src/data",
        "src/utils",
        "configs"
    ]

    # Create base directory
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True)

    # Create all folders
    for folder in folders:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder_path}")

    print(f"\nProject structure created successfully at: {base_dir.absolute()}")
    print(f"\nStructure summary:")
    print(f"  - Raw data: Full subcategories for all splits")
    print(f"  - Processed data: Main categories only (except train/Animation/Cartoon)")
    print(f"  - Models: enhanced models storage")
    print(f"  - Features: enhanced features storage")
    print(f"  - Results: output storage")


if __name__ == "__main__":
    create_project_structure()