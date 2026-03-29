# -- coding: utf-8 --

import os
import sys
import requests
import json
import progressbar
import time
import subprocess

PATH = os.path.dirname(os.path.abspath(__file__)) 

def create_folder(fd):
    '''Check and create folder for path if the folder is not exited.
    Args:
    fd: string, path of the folder
    '''
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_substring(s, begin_str, end_str):
    ''' Get substring by two given strings.
    Args: 
    s: string, orignal uncutted string
    begin_str: string, the string before the substring 
    end_str: string, the string after the substring
    Return：
    substring: the string bewteen the begin_str and end_str
    '''
    str_begin = s.find(begin_str)
    str_end = s.find(end_str)
    return s[str_begin+len(begin_str):str_end]

def parse_download_entry(line):
    """Parse a download entry line to extract category and count.
    
    Args:
        line: string, format should be 'category:count' or just 'category'
    
    Returns:
        tuple: (category_name, count) where count defaults to 10 if not specified
    """
    line = line.strip()
    if ':' in line:
        parts = line.split(':', 1)  # Split only on first colon
        category = parts[0].strip()
        try:
            count = int(parts[1].strip())
        except ValueError:
            print(f"Warning: Invalid count '{parts[1].strip()}' for category '{category}'. Using default count of 10.")
            count = 10
    else:
        category = line
        count = 10  # Default count
    
    return category, count

def check_ytdlp_installed():
    """Check if yt-dlp is installed"""
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("[OK] yt-dlp is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not found. Trying to install...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
            print("[OK] yt-dlp installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install yt-dlp automatically")
            return False

def check_pytube_available():
    """Check if pytube is available"""
    try:
        from pytube import YouTube
        from pytube.exceptions import VideoUnavailable
        return True, YouTube, VideoUnavailable
    except ImportError:
        return False, None, None

class Download():
    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))   
        self.p_bar = progressbar.ProgressBar() 
        self.categories_dict = {}
        categories_file_path = os.path.join(self.PATH,"categories.txt")     
        
        # Create state directory for tracking download progress
        self.state_dir = os.path.join(self.PATH, "download_state")
        create_folder(self.state_dir)
        
        try:
            categories_file = open(categories_file_path,"r")  
            categories_list = categories_file.readlines() 
            categories_file.close()
        except FileNotFoundError:
            print(f"Error: categories.txt not found at {categories_file_path}")
            return
        
        for categories_item in categories_list:
            try:
                self.categories = get_substring(categories_item, '\"\t', ' (')
                categories_id = get_substring(categories_item, ' \"', '\"\t')
                categories_num = get_substring(categories_item, ' (', ')')
                self.categories_dict[self.categories] = [categories_id, categories_num]
            except:
                continue

    def load_download_state(self, categories_name):
        """Load download state from JSON file"""
        state_file = os.path.join(self.state_dir, f"{categories_name}_state.json")
        default_state = {
            'target_count': 0,
            'downloaded_count': 0,
            'failed_count': 0,
            'last_processed_index': 0,
            'processed_video_ids': [],
            'downloaded_video_ids': [],
            'failed_video_ids': [],
            'exhausted_dataset': False,  # New field to track if we've reached end of dataset
            'timestamp': time.time()
        }
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # Ensure all required keys exist
                    for key in default_state:
                        if key not in state:
                            state[key] = default_state[key]
                    return state
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load state file {state_file}: {e}")
                print("Starting fresh...")
        
        return default_state

    def save_download_state(self, categories_name, state):
        """Save download state to JSON file"""
        state_file = os.path.join(self.state_dir, f"{categories_name}_state.json")
        state['timestamp'] = time.time()
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save state file {state_file}: {e}")

    def get_actual_downloaded_count(self, categories_name):
        """Count actual video files in the category directory"""
        category_dir = os.path.join(self.PATH, 'videos', categories_name)
        if not os.path.exists(category_dir):
            return 0
        
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.3gp']
        count = 0
        for file in os.listdir(category_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                count += 1
        return count

    def sync_state_with_filesystem(self, categories_name, state):
        """Synchronize state with actual downloaded files"""
        actual_count = self.get_actual_downloaded_count(categories_name)
        
        if actual_count != state['downloaded_count']:
            print(f"State mismatch detected: State shows {state['downloaded_count']}, filesystem has {actual_count}")
            print("Synchronizing with filesystem...")
            state['downloaded_count'] = actual_count
            self.save_download_state(categories_name, state)
        
        return state

    def download(self, categories_name="Maltese", max_videos=10):
        """Main download function with resume capability - keeps trying until target is met"""
        # Load previous state
        state = self.load_download_state(categories_name)
        state = self.sync_state_with_filesystem(categories_name, state)
        
        # Check if target has changed
        if state['target_count'] != max_videos:
            if max_videos < state['downloaded_count']:
                print(f"Target count ({max_videos}) is less than already downloaded ({state['downloaded_count']})")
                print("Target already exceeded!")
                return
            
            print(f"Target changed from {state['target_count']} to {max_videos}")
            state['target_count'] = max_videos
            # Reset exhausted flag if target increased
            if max_videos > state['target_count']:
                state['exhausted_dataset'] = False
            self.save_download_state(categories_name, state)
        
        # Check if we've already reached our target
        if state['downloaded_count'] >= max_videos:
            print(f"Category '{categories_name}' target already reached: {state['downloaded_count']}/{max_videos}")
            return
            
        # Check if we've exhausted the dataset without meeting target
        if state['exhausted_dataset']:
            print(f"Category '{categories_name}' dataset exhausted: {state['downloaded_count']}/{max_videos} downloaded")
            print("No more videos available in this category.")
            return

        remaining_needed = max_videos - state['downloaded_count']
        print(f"Category '{categories_name}': {state['downloaded_count']}/{max_videos} completed, need {remaining_needed} more")

        # Keep trying batches until we reach our target or exhaust the dataset
        batch_size = min(50, remaining_needed * 3)  # Get more IDs than needed to account for failures
        
        while state['downloaded_count'] < max_videos and not state['exhausted_dataset']:
            print(f"\nFetching batch of video IDs (starting from index {state['last_processed_index']})...")
            
            video_id_list = self.download_id_batch(categories_name, batch_size, state)
            
            if not video_id_list or len(video_id_list) == 0:
                print("No more video IDs available - dataset exhausted")
                state['exhausted_dataset'] = True
                self.save_download_state(categories_name, state)
                break
            
            print(f"Got {len(video_id_list)} new video IDs to try")
            
            # Download this batch
            self.download_video_batch(video_id_list, categories_name, state, max_videos)
            
            # Check if we've reached our target
            if state['downloaded_count'] >= max_videos:
                print(f"Target reached! Downloaded {state['downloaded_count']}/{max_videos}")
                break
            
            print(f"Still need {max_videos - state['downloaded_count']} more videos, continuing...")

    def download_id_batch(self, categories_name="Maltese", batch_size=50, state=None):
        """Fetch a batch of video IDs, continuing from last processed index"""
        if categories_name in self.categories_dict.keys(): 
            print("Label is Found in categories list")
            categories_id = self.categories_dict[categories_name][0]
            categories_num = self.categories_dict[categories_name][1]
        else: 
            print(categories_name + " is not present")
            return None
        
        categories_file_dir = os.path.join(self.PATH,"ID")     
        create_folder(categories_file_dir)

        try:
            video_content = requests.get('https://storage.googleapis.com/data.yt8m.org/2/j/v/'+ categories_id + '.js', timeout=30).text
        except requests.RequestException as e:
            print(f"Error fetching video content: {e}")
            return None

        find_item = '\"' + categories_id + '\",[\"'
        video_str = get_substring(video_content, find_item, '\"]')
        video_list = list(video_str.split("\",\"")) 
        
        print(f"Total videos in dataset: {len(video_list)}")
        print(f"Starting from index: {state['last_processed_index']}")
        
        if state['last_processed_index'] >= len(video_list):
            print("Reached end of dataset")
            state['exhausted_dataset'] = True
            return []
        
        video_id_list = []
        processed_video_ids = set(state['processed_video_ids'])
        
        # Calculate progress bar maximum
        max_to_process = min(len(video_list) - state['last_processed_index'], batch_size * 2)
        self.p_bar.max_value = max_to_process
        self.p_bar.start()

        successful_count = 0
        current_index = state['last_processed_index']
        p_count = 0
        
        while current_index < len(video_list) and successful_count < batch_size:
            video_item = video_list[current_index]
            
            try:
                video_response = requests.get(
                    'https://storage.googleapis.com/data.yt8m.org/2/j/i/'+ video_item[0:2] + '/' + video_item + '.js', 
                    timeout=10
                ).text
                
                if video_response[0:10] == '<?xml vers': 
                    current_index += 1
                    p_count += 1
                    self.p_bar.update(min(p_count, max_to_process))
                    continue
                    
                video_id = get_substring(video_response, "\",\"", '\")')
                if video_id and len(video_id) == 11:  # YouTube video IDs are 11 characters
                    # Skip if already processed (downloaded or failed)
                    if video_id not in processed_video_ids:
                        video_id_list.append(video_id)
                        processed_video_ids.add(video_id)
                        successful_count += 1
                        
                        # Update state periodically
                        if successful_count % 10 == 0:
                            state['last_processed_index'] = current_index + 1
                            state['processed_video_ids'] = list(processed_video_ids)
                            self.save_download_state(categories_name, state)

            except requests.RequestException:
                pass  # Skip failed requests
                
            current_index += 1
            p_count += 1
            self.p_bar.update(min(p_count, max_to_process))

        self.p_bar.finish()
        
        # Update state
        state['last_processed_index'] = current_index
        state['processed_video_ids'] = list(processed_video_ids)
        
        # Check if we've reached the end of the dataset
        if current_index >= len(video_list):
            state['exhausted_dataset'] = True
            print("Reached end of dataset")
        
        self.save_download_state(categories_name, state)
        
        print(f"Found {successful_count} new video IDs")
        print(f"Processed up to index {current_index} of {len(video_list)}")
        
        # Save/append to ID file
        if video_id_list:
            categories_file_path = os.path.join(self.PATH, categories_file_dir, f'{categories_name}.txt') 
            mode = 'a' if os.path.exists(categories_file_path) else 'w'
            with open(categories_file_path, mode, encoding='utf-8') as filehandle:
                for video_id in video_id_list:
                    filehandle.write(f'{video_id}\n')

            print(f"{'Appended' if mode == 'a' else 'Saved'} {len(video_id_list)} new IDs to {categories_file_path}")
        
        return video_id_list

    def download_video_batch(self, video_id_list, categories_name, state, target_count):
        """Download videos with state tracking - stops when target is reached"""
        print(f"Starting download of up to {len(video_id_list)} videos for {categories_name}")
        print(f"Current progress: {state['downloaded_count']}/{target_count}")
        
        # Check available download methods
        ytdlp_available = check_ytdlp_installed()
        pytube_available, YouTube, VideoUnavailable = check_pytube_available()
        
        if not ytdlp_available and not pytube_available:
            print("[ERROR] No download methods available. Please install yt-dlp or pytube.")
            return
        
        video_dir = os.path.join(self.PATH, 'videos')
        create_folder(video_dir)
        category_dir = os.path.join(video_dir, categories_name)
        create_folder(category_dir)
        
        failed_log_path = os.path.join(category_dir, 'failed_downloads.txt')
        
        self.p_bar.max_value = len(video_id_list)
        self.p_bar.start()
        
        downloads_this_batch = 0
        
        for i, video_id in enumerate(video_id_list):
            # Check if we've reached our target
            if state['downloaded_count'] >= target_count:
                print(f"\nTarget of {target_count} videos reached! Stopping download.")
                break
                
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            remaining = target_count - state['downloaded_count']
            print(f"\n[{state['downloaded_count']+1}/{target_count}] [{i+1}/{len(video_id_list)}] Downloading: {video_url}")
            print(f"Need {remaining} more videos")
            
            success = False
            error_messages = []
            
            # Method 1: Try yt-dlp first (most reliable)
            if ytdlp_available and not success:
                try:
                    print("  Trying yt-dlp...")
                    cmd = [
                        'yt-dlp',
                        '--format', 'best[height<=720]/best',
                        '--output', os.path.join(category_dir, '%(title)s.%(ext)s'),
                        '--no-playlist',
                        '--ignore-errors',
                        '--no-warnings',
                        video_url
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print("  [OK] Successfully downloaded with yt-dlp")
                        state['downloaded_count'] += 1
                        state['downloaded_video_ids'].append(video_id)
                        downloads_this_batch += 1
                        success = True
                    else:
                        error_msg = result.stderr.strip()
                        error_messages.append(f"yt-dlp: {error_msg[:100]}")
                        print(f"  yt-dlp failed: {error_msg[:100]}")
                        
                except subprocess.TimeoutExpired:
                    error_messages.append("yt-dlp: Timeout (5 minutes)")
                    print("  yt-dlp failed: Timeout")
                except Exception as e:
                    error_messages.append(f"yt-dlp: {str(e)[:100]}")
                    print(f"  yt-dlp failed: {str(e)[:100]}")
            
            # Method 2: Try pytube as fallback
            if pytube_available and not success:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"  Trying pytube (attempt {attempt + 1}/{max_retries})...")
                        
                        yt = YouTube(video_url)
                        
                        # Try different stream selection strategies
                        stream = None
                        
                        # Strategy 1: Progressive MP4 streams
                        streams = yt.streams.filter(progressive=True, file_extension='mp4')
                        if streams:
                            stream = streams.order_by('resolution').desc().first()
                        
                        # Strategy 2: Any MP4 stream
                        if not stream:
                            streams = yt.streams.filter(file_extension='mp4')
                            if streams:
                                stream = streams.first()
                        
                        # Strategy 3: Any available stream
                        if not stream:
                            stream = yt.streams.first()
                        
                        if stream:
                            print(f"    Found stream: {stream.resolution or 'unknown'} quality")
                            stream.download(category_dir)
                            print(f"  [OK] Successfully downloaded with pytube: {yt.title[:50]}")
                            state['downloaded_count'] += 1
                            state['downloaded_video_ids'].append(video_id)
                            downloads_this_batch += 1
                            success = True
                            break
                        else:
                            error_messages.append("pytube: No streams available")
                            print("    No streams available")
                            
                    except VideoUnavailable:
                        error_messages.append("pytube: Video unavailable")
                        print("    Video unavailable")
                        break
                    except Exception as e:
                        error_msg = str(e)[:100]
                        error_messages.append(f"pytube attempt {attempt+1}: {error_msg}")
                        print(f"    Attempt {attempt + 1} failed: {error_msg}")
                        
                        if attempt < max_retries - 1:
                            print("    Waiting before retry...")
                            time.sleep(3)
            
            # Log failure and update state (but continue trying other videos)
            if not success:
                print("  [FAILED] All download methods failed - will try next video")
                state['failed_count'] += 1
                state['failed_video_ids'].append(video_id)
                
                with open(failed_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{video_url}\n")
                    for error in error_messages:
                        f.write(f"  - {error}\n")
                    f.write("\n")
            
            # Save state after every download attempt
            self.save_download_state(categories_name, state)
            
            self.p_bar.update(i + 1)
            
            # Small delay between downloads to be respectful
            time.sleep(2)

        self.p_bar.finish()
        
        # Batch summary
        print(f"\nBatch Summary:")
        print(f"  Processed: {len(video_id_list)} video IDs")
        print(f"  Successfully downloaded this batch: {downloads_this_batch}")
        print(f"  Total downloaded so far: {state['downloaded_count']}/{target_count}")
        print(f"  Still need: {max(0, target_count - state['downloaded_count'])}")

    # Keep original methods for backward compatibility
    def download_id(self, categories_name="Maltese", max_videos=10):
        """Original download_id method (deprecated - use download instead)"""
        print("Warning: download_id is deprecated. Use download() method instead for resume capability.")
        return self.download_id_batch(categories_name, max_videos * 2, self.load_download_state(categories_name))

    def download_video(self, video_id_list, categories_name):
        """Original download_video method (deprecated - use download instead)"""
        print("Warning: download_video is deprecated. Use download() method instead for resume capability.")
        state = self.load_download_state(categories_name)
        return self.download_video_batch(video_id_list, categories_name, state, len(video_id_list))

def main():
    print("YouTube 8M Video Downloader - Persistent Until Target Met")
    print("="*60)
    
    labels_path = os.path.join(PATH, "downloadlist.txt") 
    try:
        with open(labels_path, "r") as labels_file:
            labels_list = labels_file.readlines()
    except FileNotFoundError:
        print(f"Error: downloadlist.txt not found at {labels_path}")
        print("Please create a downloadlist.txt file with entries in format:")
        print("  Animation:10")
        print("  Music:5")
        print("  Sports:15")
        print("  Education")
        print("(If no count is specified, defaults to 10 videos)")
        return
        
    print(f"Loaded {len(labels_list)} entries from downloadlist.txt")
    print("Format expected: 'category:count' or just 'category' (defaults to 10)")
    print("Note: Will keep trying until exact count is reached or dataset is exhausted!")
    
    download = Download()
    
    for dcount, label_entry in enumerate(labels_list):
        if not label_entry.strip():  # Skip empty lines
            continue
            
        category_name, video_count = parse_download_entry(label_entry)
        
        if not category_name:  # Skip if category name is empty
            continue
            
        print(f"\n--- Processing category {dcount + 1} of {len(labels_list)}: {category_name} ({video_count} videos) ---")
        download.download(category_name, max_videos=video_count)
        print(f"Completed category {dcount + 1} of {len(labels_list)}")
        
    print(f"\n[SUCCESS] All categories processed!")
    print(f"State files saved in: {os.path.join(PATH, 'download_state')}")
        
if __name__ == "__main__":
    main()