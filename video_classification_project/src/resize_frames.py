# resize_frames.py
# LOCATION : video_classification_project/src/
# PURPOSE  : Resize all extracted frames to 224x224
# INPUT    : video_classification_project/frames/
# OUTPUT   : video_classification_project/frames_resized/
# USAGE    : python src/resize_frames.py
#
# PREPROCESSING CONTRACT — locked forever, never change
#   FRAME_SIZE = (224, 224)   target size for all frames
#   INTERPOLATION = cv2.INTER_LINEAR   standard resize method
#
# NOTE : Output is still PNG — still visually inspectable
#        Normalization happens in the NEXT stage
#        This stage only resizes — nothing else

import os
import cv2
import json
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR       = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
FRAMES_DIR     = os.path.join(BASE_DIR, 'frames')
RESIZED_DIR    = os.path.join(BASE_DIR, 'frames_resized')
LOG_DIR        = os.path.join(BASE_DIR, 'logs')
LOG_FILE       = os.path.join(LOG_DIR,  'resize.log')
STATS_FILE     = os.path.join(LOG_DIR,  'resize_stats.json')

# ── Classes and Splits ────────────────────────────────────────────────
CLASSES = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']
SPLITS  = ['train', 'val', 'test']

# ── Preprocessing Contract ────────────────────────────────────────────
# These values are locked and used identically in
# resize, normalization, training, and inference
FRAME_SIZE    = (224, 224)          # width, height
INTERPOLATION = cv2.INTER_LINEAR    # standard interpolation

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs(RESIZED_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)
open(LOG_FILE, 'w').close()

# ── Logging ───────────────────────────────────────────────────────────
def log(msg, level='INFO'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg  = f"[{timestamp}] [{level}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

# ── Resize Single Frame ───────────────────────────────────────────────
def resize_frame(frame):
    """
    Resizes a single frame to FRAME_SIZE using INTER_LINEAR.
    Input  : numpy array of any size (BGR)
    Output : numpy array of shape (224, 224, 3) (BGR)
    """
    return cv2.resize(frame, FRAME_SIZE, interpolation=INTERPOLATION)

# ── Count Total Frames ────────────────────────────────────────────────
def count_total_frames():
    """
    Counts total PNG files to process for progress tracking.
    """
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = os.path.join(FRAMES_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for sub in os.listdir(cls_dir):
                sub_path = os.path.join(cls_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                for vid in os.listdir(sub_path):
                    vid_path = os.path.join(sub_path, vid)
                    if not os.path.isdir(vid_path):
                        continue
                    total += len([
                        f for f in os.listdir(vid_path)
                        if f.endswith('.png')])
    return total

# ── Verify Output ─────────────────────────────────────────────────────
def verify_output():
    """
    After resize, confirms all output frames are correct size.
    Samples one frame per class per split for verification.
    """
    log("\n" + "─" * 60)
    log("OUTPUT VERIFICATION")
    log("─" * 60)

    all_passed = True

    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = os.path.join(RESIZED_DIR, split, cls)
            if not os.path.exists(cls_dir):
                log(f"  MISSING: {split}/{cls}", 'WARN')
                all_passed = False
                continue

            # Find first available frame
            sample_frame = None
            for sub in os.listdir(cls_dir):
                sub_path = os.path.join(cls_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                for vid in os.listdir(sub_path):
                    vid_path = os.path.join(sub_path, vid)
                    if not os.path.isdir(vid_path):
                        continue
                    frames = [
                        f for f in os.listdir(vid_path)
                        if f.endswith('.png')]
                    if frames:
                        sample_frame = os.path.join(
                            vid_path, frames[0])
                        break
                if sample_frame:
                    break

            if sample_frame is None:
                log(f"  NO FRAMES: {split}/{cls}", 'WARN')
                all_passed = False
                continue

            # Check shape
            frame = cv2.imread(sample_frame)
            if frame is None:
                log(f"  UNREADABLE: {split}/{cls}", 'WARN')
                all_passed = False
                continue

            h, w, c = frame.shape
            expected_h = FRAME_SIZE[1]
            expected_w = FRAME_SIZE[0]

            if h == expected_h and w == expected_w and c == 3:
                log(f"  {split}/{cls:<20} "
                    f"shape: ({h},{w},{c}) PASS")
            else:
                log(f"  {split}/{cls:<20} "
                    f"shape: ({h},{w},{c}) "
                    f"expected ({expected_h},{expected_w},3) FAIL",
                    'WARN')
                all_passed = False

    if all_passed:
        log("\nALL SHAPE CHECKS PASSED")
    else:
        log("\nSOME SHAPE CHECKS FAILED — investigate before proceeding",
            'WARN')

    return all_passed

# ── Main Resize ───────────────────────────────────────────────────────
def run_resize():
    log("=" * 60)
    log("FRAME RESIZE")
    log("=" * 60)
    log(f"Input          : {FRAMES_DIR}")
    log(f"Output         : {RESIZED_DIR}")
    log(f"Target size    : {FRAME_SIZE}")
    log(f"Interpolation  : INTER_LINEAR")
    log("=" * 60)

    # Count total frames for progress
    log("\nCounting total frames...")
    total_frames = count_total_frames()
    log(f"Total frames to resize: {total_frames}")

    # Stats
    stats = {
        'config': {
            'frame_size'   : FRAME_SIZE,
            'interpolation': 'INTER_LINEAR'
        },
        'results': {},
        'totals' : {
            'processed': 0,
            'success'  : 0,
            'failed'   : 0,
        }
    }

    processed  = 0
    start_time = datetime.now()

    for split in SPLITS:
        log(f"\n{'#'*60}")
        log(f"SPLIT: {split.upper()}")
        log(f"{'#'*60}")

        stats['results'][split] = {}

        for cls in CLASSES:
            src_cls = os.path.join(FRAMES_DIR,  split, cls)
            dst_cls = os.path.join(RESIZED_DIR, split, cls)

            log(f"\n{'─'*50}")
            log(f"{split}/{cls}")

            if not os.path.exists(src_cls):
                log(f"SKIP: source not found", 'WARN')
                continue

            cls_success = 0
            cls_failed  = 0

            subcategories = sorted([
                d for d in os.listdir(src_cls)
                if os.path.isdir(os.path.join(src_cls, d))
            ])

            for subcategory in subcategories:
                src_sub = os.path.join(src_cls, subcategory)
                dst_sub = os.path.join(dst_cls, subcategory)

                video_dirs = sorted([
                    d for d in os.listdir(src_sub)
                    if os.path.isdir(
                        os.path.join(src_sub, d))
                ])

                for video_dir in video_dirs:
                    src_vid = os.path.join(src_sub, video_dir)
                    dst_vid = os.path.join(dst_sub, video_dir)
                    os.makedirs(dst_vid, exist_ok=True)

                    frame_files = sorted([
                        f for f in os.listdir(src_vid)
                        if f.endswith('.png')
                    ])

                    for frame_file in frame_files:
                        src_path = os.path.join(
                            src_vid, frame_file)
                        dst_path = os.path.join(
                            dst_vid, frame_file)

                        # Read
                        frame = cv2.imread(src_path)

                        if frame is None:
                            log(f"    FAIL_READ: {src_path}",
                                'WARN')
                            cls_failed += 1
                            stats['totals']['failed'] += 1
                            processed += 1
                            continue

                        # Resize
                        resized = resize_frame(frame)

                        # Save
                        cv2.imwrite(dst_path, resized)

                        cls_success += 1
                        stats['totals']['success'] += 1
                        processed += 1
                        stats['totals']['processed'] += 1

                        # Progress every 1000 frames
                        if processed % 1000 == 0:
                            elapsed = (datetime.now() -
                                       start_time).seconds
                            pct     = processed / total_frames * 100
                            log(f"  ── Progress: {processed}/"
                                f"{total_frames} "
                                f"({pct:.1f}%) | "
                                f"Elapsed: {elapsed//60}m "
                                f"{elapsed%60}s")

            log(f"  {split}/{cls} DONE: "
                f"success={cls_success} failed={cls_failed}")

            stats['results'][split][cls] = {
                'success': cls_success,
                'failed' : cls_failed
            }

    # Final summary
    elapsed_total = (datetime.now() - start_time).seconds

    log("\n" + "=" * 60)
    log("RESIZE COMPLETE")
    log("=" * 60)
    log(f"Total time  : {elapsed_total//60}m {elapsed_total%60}s")
    log(f"Processed   : {stats['totals']['processed']}")
    log(f"Success     : {stats['totals']['success']}")
    log(f"Failed      : {stats['totals']['failed']}")
    log(f"\nOutput      : {RESIZED_DIR}")
    log(f"Log file    : {LOG_FILE}")

    # Save stats
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    log(f"Stats saved : {STATS_FILE}")

    # Verify output
    verify_output()

    log("\n" + "─" * 60)
    log("NEXT STEPS:")
    log("─" * 60)
    log("1. Run resize verification:")
    log("   python verification/verify_resize.py")
    log("2. Only proceed to normalization after verification passes")
    log("─" * 60)


run_resize()
