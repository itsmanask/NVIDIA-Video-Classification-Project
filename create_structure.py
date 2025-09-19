import os
from pathlib import Path

def create_project_structure(base_path="video_classification_project"):
    """
    Create the full folder structure for the video classification project
    based on the current dataset organization.
    """

    # Dataset categories and subcategories
    categories = {
        "Animation": [
            "Animation", "Bleach", "Cartoon", "Dragon Ball",
            "Lego minifigure", "Mickey Mouse", "Naruto",
            "One Piece", "Sonic the Hedgehog",
            "The Walt Disney Company", "Walt Disney World"
        ],
        "Flat_Content": [
            "Chart", "Illustration", "Logo", "Map",
            "Poster", "Screencast", "Text",
            "Typography", "Website"
        ],
        "Gaming": [
            "Action-adventure game", "Battlefield", "Call of Duty",
            "Counter-Strike", "FIFA 15", "Games",
            "Grand Theft Auto", "Grand Theft Auto V",
            "League of Legends", "Minecraft", "Need for Speed",
            "RuneScape", "Video game", "World of Warcraft"
        ],
        "Natural_Content": [
            "Animal", "Bear", "Bird", "Cat", "Chicken",
            "Deer", "Dog", "Elephant", "Farm", "Fish",
            "Fishing", "Garden", "Horse", "Lion",
            "Nature", "Outdoor recreation", "Pet",
            "Plant", "Tree", "Wildlife"
        ]
    }

    # Base dataset splits
    splits = ["train", "val", "test"]

    # Define folders
    folders = []

    # Raw + Processed data splits
    for split in splits:
        for category, subcats in categories.items():
            for sub in subcats:
                folders.append(f"data/raw/{split}/{category}/{sub}")
                folders.append(f"data/processed/{split}/{category}/{sub}")

    # Project support folders (notebooks & results removed)
    folders += [
        "models/checkpoints",
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


if __name__ == "__main__":
    create_project_structure()
