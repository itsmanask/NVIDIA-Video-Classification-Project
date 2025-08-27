import os
from pathlib import Path

def analyze_dataset(base_path):
    """
    Analyze your current dataset structure and count videos
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Directory {base_path} does not exist!")
        return
    
    print(f"Analyzing dataset at: {base_path}")
    print("=" * 60)
    
    categories = ['Animation', 'Gaming', 'Natural Content', 'Flat Content']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    total_videos = 0
    category_counts = {}
    
    for category in categories:
        category_path = base_dir / category / 'videos'
        
        if category_path.exists():
            # Count video files
            video_count = 0
            video_files = []
            
            for ext in video_extensions:
                pattern = f"*{ext}"
                files = list(category_path.glob(pattern))
                video_files.extend(files)
                video_count += len(files)
            
            category_counts[category] = video_count
            total_videos += video_count
            
            print(f"{category:20} | {video_count:4} videos | Path: {category_path}")
            
            # Show a few example filenames
            if video_files:
                print(f"  Example files:")
                for i, file in enumerate(video_files[:3]):
                    print(f"    - {file.name}")
                if len(video_files) > 3:
                    print(f"    ... and {len(video_files) - 3} more")
            print()
        else:
            print(f"{category:20} | NOT FOUND | Expected: {category_path}")
            category_counts[category] = 0
    
    print("=" * 60)
    print(f"TOTAL VIDEOS: {total_videos}")
    print()
    
    # Show distribution percentages
    if total_videos > 0:
        print("Category Distribution:")
        for category, count in category_counts.items():
            percentage = (count / total_videos) * 100
            print(f"  {category:20}: {count:4} videos ({percentage:5.1f}%)")
    
    return category_counts, total_videos

def check_video_properties(base_path, sample_size=5):
    """
    Check properties of a few sample videos
    """
    import cv2
    
    base_dir = Path(base_path)
    categories = ['Animation', 'Gaming', 'Natural Content', 'Flat Content']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    print("\nChecking sample video properties...")
    print("=" * 60)
    
    for category in categories:
        category_path = base_dir / category / 'videos'
        
        if not category_path.exists():
            continue
            
        # Get sample videos
        video_files = []
        for ext in video_extensions:
            pattern = f"*{ext}"
            files = list(category_path.glob(pattern))
            video_files.extend(files)
        
        if not video_files:
            continue
            
        print(f"\n{category} - Sample Video Properties:")
        
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
                    
                    print(f"  {video_file.name[:40]:40} | {width}x{height} | {fps:5.1f}fps | {duration:5.1f}s | {frame_count} frames")
                    cap.release()
                else:
                    print(f"  {video_file.name[:40]:40} | ERROR: Could not open")
            except Exception as e:
                print(f"  {video_file.name[:40]:40} | ERROR: {str(e)}")

if __name__ == "__main__":
    # Your dataset path
    dataset_path = r"D:\PR1\Dataset"
    
    print("Dataset Analysis")
    print("================")
    
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
        print("1. The path D:\\PR1\\Dataset exists")
        print("2. It contains folders: Animation, Gaming, 'Natural Content', 'Flat Content'")
        print("3. Each folder has a 'videos' subfolder with video files")