import os
from pathlib import Path

def analyze_dataset(base_path):
    """
    Analyze your current dataset structure with subcategories
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Directory {base_path} does not exist!")
        return
    
    print(f"Analyzing dataset at: {base_path}")
    print("=" * 80)
    
    # Define all subcategories for each main category
    category_structure = {
        'Animation': [
            'Animation', 'Bleach', 'Cartoon', 'Dragon Ball',
            'Lego minifigure', 'Mickey Mouse', 'Naruto',
            'One Piece', 'Sonic the Hedgehog',
            'The Walt Disney Company', 'Walt Disney World'
        ],
        
        'Gaming': [
            'Action-adventure game', 'Battlefield', 'Call of Duty',
            'Counter-Strike', 'FIFA 15', 'Games',
            'Grand Theft Auto', 'Grand Theft Auto V',
            'League of Legends', 'Minecraft', 'Need for Speed',
            'RuneScape', 'Video game', 'World of Warcraft'
        ],
        
        'Natural Content': [
            'Animal', 'Bear', 'Bird', 'Cat', 'Chicken',
            'Deer', 'Dog', 'Elephant', 'Farm', 'Fish',
            'Fishing', 'Garden', 'Horse', 'Lion',
            'Nature', 'Outdoor recreation', 'Pet',
            'Plant', 'Tree', 'Wildlife'
        ],
        
        'Flat Content': [
            'Chart', 'Illustration', 'Logo', 'Map',
            'Poster', 'Screencast', 'Text',
            'Typography', 'Website'
        ]
    }

    
    # Use case-insensitive extension matching to avoid duplicates
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}  # Use set for faster lookup
    
    total_videos = 0
    category_totals = {}
    all_subcategory_counts = {}
    
    for main_category, subcategories in category_structure.items():
        print(f"\n{main_category.upper()}:")
        print("-" * (len(main_category) + 1))
        
        category_total = 0
        category_totals[main_category] = 0
        
        for subcategory in subcategories:
            subcategory_path = base_dir / main_category / 'videos' / subcategory
            
            if subcategory_path.exists():
                # Get all files and filter by video extensions (case-insensitive)
                all_files = list(subcategory_path.iterdir())
                video_files = [
                    f for f in all_files 
                    if f.is_file() and f.suffix.lower() in video_extensions
                ]
                
                video_count = len(video_files)
                
                category_total += video_count
                all_subcategory_counts[f"{main_category}/{subcategory}"] = video_count
                
                status = "✓" if video_count > 0 else "✗"
                print(f"  {status} {subcategory:25} | {video_count:4} videos | {subcategory_path}")
                
                # Show example files for subcategories with videos
                if video_count > 0 and len(video_files) >= 3:
                    print(f"    Examples: {', '.join([f.name for f in video_files[:3]])}")
            else:
                all_subcategory_counts[f"{main_category}/{subcategory}"] = 0
                print(f"  ✗ {subcategory:25} | NOT FOUND | {subcategory_path}")
        
        category_totals[main_category] = category_total
        total_videos += category_total
        print(f"  → {main_category} TOTAL: {category_total} videos")
    
    print("\n" + "=" * 80)
    print(f"GRAND TOTAL: {total_videos} videos")
    print()
    
    # Show main category distribution
    if total_videos > 0:
        print("Main Category Distribution:")
        for category, count in category_totals.items():
            percentage = (count / total_videos) * 100
            print(f"  {category:20}: {count:4} videos ({percentage:5.1f}%)")
        
        print(f"\nTop 10 Subcategories by Video Count:")
        sorted_subcats = sorted(all_subcategory_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for subcat, count in sorted_subcats:
            if count > 0:
                print(f"  {subcat:35}: {count:4} videos")
    
    return category_totals, total_videos

def check_video_properties(base_path, sample_size=2):
    """
    Check properties of a few sample videos from each subcategory
    """
    import cv2
    
    base_dir = Path(base_path)
    
    # Define subcategories to sample from (just the major ones)
    sample_subcategories = {
        'Animation': ['Cartoon', 'Animation', 'Lego minifigure'],
        'Gaming': ['Games', 'Video game', 'Minecraft'], 
        'Natural Content': ['Animal', 'Pet', 'Dog'],
        'Flat Content': ['Website', 'Chart', 'Map']
    }
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    print("\nChecking sample video properties...")
    print("=" * 80)
    
    for main_category, subcategories in sample_subcategories.items():
        print(f"\n{main_category} - Sample Video Properties:")
        
        for subcategory in subcategories:
            subcategory_path = base_dir / main_category / 'videos' / subcategory
            
            if not subcategory_path.exists():
                continue
                
            # Get video files (case-insensitive)
            all_files = list(subcategory_path.iterdir())
            video_files = [
                f for f in all_files 
                if f.is_file() and f.suffix.lower() in video_extensions
            ]
            
            if not video_files:
                continue
                
            print(f"\n  {subcategory}:")
            sample_files = video_files[:sample_size]
            
            for video_file in sample_files:
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        print(f"    {video_file.name[:35]:35} | {width}x{height} | {fps:5.1f}fps | {duration:5.1f}s")
                        cap.release()
                    else:
                        print(f"    {video_file.name[:35]:35} | ERROR: Could not open")
                except Exception as e:
                    print(f"    {video_file.name[:35]:35} | ERROR: {str(e)}")

if __name__ == "__main__":
    # Use relative path from current working directory (PR1)
    dataset_path = "Dataset"  # Assumes you're running from PR1 directory
    
    print("Dataset Analysis")
    print("================")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for dataset at: {Path(dataset_path).absolute()}")
    print()
    
    try:
        category_counts, total_videos = analyze_dataset(dataset_path)
        
        if total_videos > 0:
            # Check a few sample videos
            check_video_properties(dataset_path, sample_size=3)
        else:
            print("No videos found. Please check your directory structure.")
            
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        print("\nPlease ensure:")
        print("1. You are running this script from the PR1 directory")
        print("2. The Dataset folder exists in the current directory")
        print("3. It contains folders: Animation, Gaming, 'Natural Content', 'Flat Content'")
        print("4. Each folder has: Category/videos/Subcategory/ structure")
        print("   Example: Dataset/Animation/videos/Cartoon/")
        print("   Example: Dataset/Gaming/videos/Games/")