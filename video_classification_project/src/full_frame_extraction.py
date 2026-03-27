# full_frame_extraction.py
# LOCATION : video_classification_project/src/
# PURPOSE  : Extract frames from entire dataset (train + val + test)
# OUTPUT   : video_classification_project/frames/
# USAGE    : python full_frame_extraction.py
#
# PREPROCESSING CONTRACT — locked forever, never change
#   NUM_FRAMES   = 16     final frames per video
#   SKIP_EDGES   = 0.1    skip first and last 10%
#   BLACK_THRESH = 25     mean pixel below this = black frame
#   N_CANDIDATES = 48     extracted before filtering
#
# NOTE: Logo, Illustration, Typography already deleted from raw data
#       No exclusion logic needed in this script

import os
import cv2
import warnings
import logging
import numpy as np
import json
from datetime import datetime

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR   = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
RAW_DIR    = os.path.join(BASE_DIR, 'data/raw')
FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
LOG_DIR    = os.path.join(BASE_DIR, 'logs')
LOG_FILE   = os.path.join(LOG_DIR,  'full_extraction.log')
STATS_FILE = os.path.join(LOG_DIR,  'extraction_stats.json')

# ── Classes and Splits ────────────────────────────────────────────────
CLASSES = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']
SPLITS  = ['train', 'val', 'test']

# ── Preprocessing Contract ────────────────────────────────────────────
NUM_FRAMES   = 16
SKIP_EDGES   = 0.1
BLACK_THRESH = 25
N_CANDIDATES = NUM_FRAMES * 3   # 48 candidates → filter → 16 final

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
open(LOG_FILE, 'w').close()

# ── Logging ───────────────────────────────────────────────────────────
def log(msg, level='INFO'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg  = f"[{timestamp}] [{level}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

# ── Black Frame Check ─────────────────────────────────────────────────
def is_black_frame(frame):
    """
    Returns True if frame is black or near-black.
    Threshold 25 catches pure black and dark fade transitions.
    """
    return frame.mean() < BLACK_THRESH

# ── Core Frame Extraction ─────────────────────────────────────────────
def extract_frames(video_path):
    """
    THE only frame extraction function in the entire project.
    Identical logic to test_frame_extraction.py.
    Never modify this function independently.

    Returns:
        frames : list of numpy arrays (BGR) — length up to NUM_FRAMES
        info   : dict with video metadata
        status : one of the following strings
                 'success'        — exactly NUM_FRAMES good frames
                 'warn_few_good'  — fewer than NUM_FRAMES good frames
                 'warn_all_black' — no good frames, used fallback
                 'skip_short'     — video too short to extract from
                 'fail_open'      — could not open video file
                 'fail_read'      — opened but could not read frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [], {}, 'fail_open'

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps if fps > 0 else 0

    info = {
        'total_frames' : total,
        'fps'          : round(fps, 2),
        'width'        : w,
        'height'       : h,
        'duration_sec' : round(dur, 2)
    }

    # Skip videos too short to sample from
    if total < NUM_FRAMES:
        cap.release()
        return [], info, 'skip_short'

    # Step 1: Calculate 48 candidate positions
    # Skip first and last 10% to avoid intros/outros/black frames
    start   = int(SKIP_EDGES * total)
    end     = int((1 - SKIP_EDGES) * total)
    indices = [
        int(start + i * (end - start) / N_CANDIDATES)
        for i in range(N_CANDIDATES)
    ]

    # Step 2: Extract candidate frames
    candidates = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            candidates.append((idx, frame))
    cap.release()

    if not candidates:
        return [], info, 'fail_read'

    # Step 3: Filter out black frames
    good = [(idx, f) for idx, f in candidates
            if not is_black_frame(f)]

    # Step 4: Select 16 evenly spread from good frames
    if len(good) >= NUM_FRAMES:
        step     = len(good) / NUM_FRAMES
        selected = [good[int(i * step)][1]
                    for i in range(NUM_FRAMES)]
        status   = 'success'

    elif len(good) > 0:
        selected = [f for _, f in good]
        status   = 'warn_few_good'

    else:
        # All frames dark — use raw candidates as last resort
        selected = [f for _, f in candidates[:NUM_FRAMES]]
        status   = 'warn_all_black'

    return selected, info, status

# ── Save Frames ───────────────────────────────────────────────────────
def save_frames(frames, split, cls, subcategory, video_name):
    """
    Saves extracted frames as PNG files.

    Output structure:
    frames/
    └── split/
        └── cls/
            └── subcategory/
                └── video_name/
                    ├── frame_00.png
                    ├── frame_01.png
                    └── ...
    """
    # Clean video name for folder name
    safe_name = "".join(
        c if c.isalnum() or c in ('_', '-') else '_'
        for c in os.path.splitext(video_name)[0]
    )[:80]

    save_dir = os.path.join(
        FRAMES_DIR, split, cls, subcategory, safe_name)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    for i, frame in enumerate(frames):
        if frame is not None:
            cv2.imwrite(
                os.path.join(save_dir, f"frame_{i:02d}.png"),
                frame)
            saved += 1

    return save_dir, saved

# ── Count Total Videos ────────────────────────────────────────────────
def count_total_videos():
    """
    Counts total videos across all splits and classes.
    Used for progress percentage tracking during extraction.
    """
    total     = 0
    breakdown = {}

    for split in SPLITS:
        breakdown[split] = {}
        for cls in CLASSES:
            cls_dir = os.path.join(RAW_DIR, split, cls)
            if not os.path.exists(cls_dir):
                breakdown[split][cls] = 0
                continue

            count = 0
            for subcategory in os.listdir(cls_dir):
                sub_path = os.path.join(cls_dir, subcategory)
                if not os.path.isdir(sub_path):
                    continue
                for f in os.listdir(sub_path):
                    if f.lower().endswith(
                            ('.mp4', '.avi', '.mov', '.mkv')):
                        count += 1

            breakdown[split][cls] = count
            total += count

    return total, breakdown

# ── Verify Output Structure ───────────────────────────────────────────
def verify_output_structure():
    """
    After extraction, counts video folders in output directory.
    Each video should have exactly one folder with 16 PNG files.
    """
    log("\n" + "─" * 60)
    log("OUTPUT STRUCTURE VERIFICATION")
    log("─" * 60)

    total_video_folders = 0
    total_frames_saved  = 0

    for split in SPLITS:
        for cls in CLASSES:
            out_dir = os.path.join(FRAMES_DIR, split, cls)

            if not os.path.exists(out_dir):
                log(f"  {split}/{cls:<20} MISSING", 'WARN')
                continue

            video_folders  = 0
            frames_in_cls  = 0

            for sub in sorted(os.listdir(out_dir)):
                sub_path = os.path.join(out_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                for vfolder in os.listdir(sub_path):
                    vpath = os.path.join(sub_path, vfolder)
                    if os.path.isdir(vpath):
                        frame_count = len([
                            f for f in os.listdir(vpath)
                            if f.endswith('.png')])
                        video_folders  += 1
                        frames_in_cls  += frame_count

            total_video_folders += video_folders
            total_frames_saved  += frames_in_cls

            avg_frames = (frames_in_cls / video_folders
                         if video_folders > 0 else 0)

            log(f"  {split}/{cls:<20} "
                f"videos: {video_folders:<6} "
                f"frames: {frames_in_cls:<8} "
                f"avg: {avg_frames:.1f}")

    log(f"\n  TOTAL video folders : {total_video_folders}")
    log(f"  TOTAL frames saved  : {total_frames_saved}")
    log(f"  Expected avg frames : {NUM_FRAMES} per video")

# ── Main Extraction ───────────────────────────────────────────────────
def run_extraction():
    log("=" * 60)
    log("FULL DATASET FRAME EXTRACTION")
    log("=" * 60)
    log(f"Base dir     : {BASE_DIR}")
    log(f"Raw data     : {RAW_DIR}")
    log(f"Output       : {FRAMES_DIR}")
    log(f"Splits       : {SPLITS}")
    log(f"Classes      : {CLASSES}")
    log(f"Frames       : {NUM_FRAMES} per video")
    log(f"Candidates   : {N_CANDIDATES} before filtering")
    log(f"Edge skip    : {SKIP_EDGES*100:.0f}%")
    log(f"Black thresh : {BLACK_THRESH}")
    log("=" * 60)

    # Count total videos for progress tracking
    log("\nCounting total videos...")
    total_videos, breakdown = count_total_videos()
    log(f"Total videos to process: {total_videos}")
    log("\nBreakdown per split/class:")
    for split in SPLITS:
        for cls in CLASSES:
            count = breakdown[split].get(cls, 0)
            log(f"  {split}/{cls:<20}: {count}")

    # ── Stats tracking ────────────────────────────────────────────────
    stats = {
        'config': {
            'num_frames'  : NUM_FRAMES,
            'skip_edges'  : SKIP_EDGES,
            'black_thresh': BLACK_THRESH,
            'n_candidates': N_CANDIDATES,
        },
        'results' : {},
        'totals'  : {
            'processed'     : 0,
            'success'       : 0,
            'warn_few_good' : 0,
            'warn_all_black': 0,
            'skip_short'    : 0,
            'fail_open'     : 0,
            'fail_read'     : 0,
        }
    }

    processed  = 0
    start_time = datetime.now()

    # ── Main Loop ─────────────────────────────────────────────────────
    for split in SPLITS:
        log(f"\n{'#'*60}")
        log(f"SPLIT: {split.upper()}")
        log(f"{'#'*60}")

        stats['results'][split] = {}

        for cls in CLASSES:
            cls_dir = os.path.join(RAW_DIR, split, cls)

            log(f"\n{'─'*50}")
            log(f"{split}/{cls}")

            if not os.path.exists(cls_dir):
                log(f"SKIP: folder not found", 'WARN')
                continue

            cls_stats = {
                'success'       : 0,
                'warn_few_good' : 0,
                'warn_all_black': 0,
                'skip_short'    : 0,
                'fail_open'     : 0,
                'fail_read'     : 0,
            }

            subcategories = sorted([
                d for d in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, d))
            ])

            log(f"Subcategories: {subcategories}")

            for subcategory in subcategories:
                sub_path = os.path.join(cls_dir, subcategory)

                videos = sorted([
                    f for f in os.listdir(sub_path)
                    if f.lower().endswith(
                        ('.mp4', '.avi', '.mov', '.mkv'))
                ])

                log(f"  {subcategory}: {len(videos)} videos")

                for video_name in videos:
                    video_path = os.path.join(sub_path, video_name)
                    processed += 1
                    stats['totals']['processed'] += 1

                    # Progress update every 50 videos
                    if processed % 50 == 0:
                        elapsed = (datetime.now() -
                                   start_time).seconds
                        pct     = processed / total_videos * 100
                        log(f"  ── Progress: {processed}/"
                            f"{total_videos} "
                            f"({pct:.1f}%) | "
                            f"Elapsed: {elapsed//60}m "
                            f"{elapsed%60}s")

                    # Extract frames
                    frames, info, status = extract_frames(
                        video_path)

                    # Track in stats
                    cls_stats[status] = \
                        cls_stats.get(status, 0) + 1
                    stats['totals'][status] = \
                        stats['totals'].get(status, 0) + 1

                    # Safe printable name for logs
                    safe_name = video_name.encode(
                        'ascii', 'replace').decode('ascii')[:50]

                    # Log only problems — not every success
                    if status == 'fail_open':
                        log(f"    FAIL_OPEN   : {safe_name}",
                            'WARN')
                        continue

                    if status == 'fail_read':
                        log(f"    FAIL_READ   : {safe_name}",
                            'WARN')
                        continue

                    if status == 'skip_short':
                        log(f"    SKIP_SHORT  : {safe_name} "
                            f"({info.get('total_frames',0)} frames)",
                            'WARN')
                        continue

                    if status == 'warn_all_black':
                        log(f"    WARN_ALL_BLACK: {safe_name}",
                            'WARN')

                    if status == 'warn_few_good':
                        log(f"    WARN_FEW_GOOD : {safe_name} "
                            f"({len(frames)} frames)",
                            'WARN')

                    # Save frames to disk
                    if frames:
                        save_frames(
                            frames, split, cls,
                            subcategory, video_name)

            # Per class summary
            log(f"\n  {split}/{cls} DONE:")
            log(f"    success       : {cls_stats['success']}")
            log(f"    warn_few_good : {cls_stats['warn_few_good']}")
            log(f"    warn_all_black: {cls_stats['warn_all_black']}")
            log(f"    skip_short    : {cls_stats['skip_short']}")
            log(f"    fail_open     : {cls_stats['fail_open']}")
            log(f"    fail_read     : {cls_stats['fail_read']}")

            stats['results'][split][cls] = cls_stats

    # ── Final Summary ─────────────────────────────────────────────────
    elapsed_total = (datetime.now() - start_time).seconds

    log("\n" + "=" * 60)
    log("EXTRACTION COMPLETE")
    log("=" * 60)
    log(f"Total time    : {elapsed_total//60}m {elapsed_total%60}s")
    log(f"Processed     : {stats['totals']['processed']}")
    log(f"Success       : {stats['totals']['success']}")
    log(f"Warn few good : {stats['totals']['warn_few_good']}")
    log(f"Warn all black: {stats['totals']['warn_all_black']}")
    log(f"Skip short    : {stats['totals']['skip_short']}")
    log(f"Fail open     : {stats['totals']['fail_open']}")
    log(f"Fail read     : {stats['totals']['fail_read']}")
    log(f"\nFrames dir    : {FRAMES_DIR}")
    log(f"Log file      : {LOG_FILE}")
    log(f"Stats file    : {STATS_FILE}")

    # Save stats as JSON
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    log(f"Stats saved to: {STATS_FILE}")

    # Verify output structure
    verify_output_structure()

    # ── Next steps ────────────────────────────────────────────────────
    log("\n" + "─" * 60)
    log("NEXT STEPS:")
    log("─" * 60)
    log("1. Check for warnings and failures:")
    log("   grep WARN logs/full_extraction.log | head -50")
    log("   grep FAIL logs/full_extraction.log | head -50")
    log("2. Check stats summary:")
    log("   cat logs/extraction_stats.json")
    log("3. Run temporal overlap check on full extracted frames:")
    log("   python src/check_temporal_overlap.py")
    log("4. Run static overlap check on full extracted frames:")
    log("   python src/check_static_overlap.py")
    log("5. Only proceed to resize after both checks pass")
    log("─" * 60)


run_extraction()
