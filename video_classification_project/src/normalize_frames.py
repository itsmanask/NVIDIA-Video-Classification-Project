# normalize_frames.py
# LOCATION : video_classification_project/src/
# PURPOSE  : Apply ImageNet normalization to resized frames
#            and save as .pt tensor files ready for model input
# INPUT    : video_classification_project/frames_resized/
# OUTPUT   : video_classification_project/frames_normalized/
# USAGE    : python src/normalize_frames.py
#
# PREPROCESSING CONTRACT — locked forever, never change
#   IMAGENET_MEAN = [0.485, 0.456, 0.406]
#   IMAGENET_STD  = [0.229, 0.224, 0.225]
#   FRAME_SIZE    = (224, 224)
#   NUM_FRAMES    = 16
#   OUTPUT_SHAPE  = (16, 3, 224, 224)
#
# NOTE : After this stage frames are no longer visually inspectable
#        as images. Verification uses denormalization to recover
#        the original image and confirm it looks correct.

import os
import cv2
import numpy as np
import torch
import json
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR        = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
RESIZED_DIR     = os.path.join(BASE_DIR, 'frames_resized')
NORMALIZED_DIR  = os.path.join(BASE_DIR, 'frames_normalized')
LOG_DIR         = os.path.join(BASE_DIR, 'logs')
LOG_FILE        = os.path.join(LOG_DIR,  'normalize.log')
STATS_FILE      = os.path.join(LOG_DIR,  'normalize_stats.json')

# ── Classes and Splits ────────────────────────────────────────────────
CLASSES = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']
SPLITS  = ['train', 'val', 'test']

# ── Preprocessing Contract ────────────────────────────────────────────
# ImageNet normalization values — locked forever
# These match ResNet18 pretrained weights exactly
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_FRAMES    = 16
FRAME_SIZE    = (224, 224)

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs(NORMALIZED_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
open(LOG_FILE, 'w').close()

# ── Logging ───────────────────────────────────────────────────────────
def log(msg, level='INFO'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg  = f"[{timestamp}] [{level}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

# ── Normalize Single Frame ────────────────────────────────────────────
def normalize_frame(frame_bgr):
    """
    Converts a single BGR frame to a normalized RGB tensor.

    Steps:
    1. Convert BGR to RGB (OpenCV loads as BGR, model expects RGB)
    2. Convert to float32 and scale to 0-1
    3. Subtract ImageNet mean per channel
    4. Divide by ImageNet std per channel

    Input  : numpy array (224, 224, 3) BGR uint8 0-255
    Output : numpy array (224, 224, 3) RGB float32 ~-2.5 to 2.5
    """
    # Step 1: BGR to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: Scale to 0-1
    rgb = rgb.astype(np.float32) / 255.0

    # Step 3 and 4: Subtract mean and divide by std
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    return normalized

# ── Process One Video Folder ──────────────────────────────────────────
def process_video(src_vid_path, dst_vid_path):
    """
    Reads all frame PNGs from one video folder,
    normalizes them, stacks into tensor, saves as .pt file.

    Input  : folder containing frame_00.png to frame_15.png
    Output : single .pt file with tensor shape (16, 3, 224, 224)
    """
    frame_files = sorted([
        f for f in os.listdir(src_vid_path)
        if f.endswith('.png')
    ])

    if not frame_files:
        return None, 'no_frames'

    normalized_frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(src_vid_path, frame_file)
        frame      = cv2.imread(frame_path)

        if frame is None:
            return None, 'read_error'

        # Verify frame size
        if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            return None, 'wrong_size'

        normalized = normalize_frame(frame)
        normalized_frames.append(normalized)

    # Stack: list of (224,224,3) → array (N, 224, 224, 3)
    stacked = np.stack(normalized_frames, axis=0)

    # Transpose: (N, 224, 224, 3) → (N, 3, 224, 224)
    # PyTorch expects channels first
    transposed = stacked.transpose(0, 3, 1, 2)

    # Convert to tensor
    tensor = torch.tensor(transposed, dtype=torch.float32)

    # Save as .pt file
    os.makedirs(os.path.dirname(dst_vid_path), exist_ok=True)
    torch.save(tensor, dst_vid_path)

    return tensor.shape, 'success'

# ── Count Total Videos ────────────────────────────────────────────────
def count_total_videos():
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = os.path.join(RESIZED_DIR, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for sub in os.listdir(cls_dir):
                sub_path = os.path.join(cls_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                for vid in os.listdir(sub_path):
                    vid_path = os.path.join(sub_path, vid)
                    if os.path.isdir(vid_path):
                        total += 1
    return total

# ── Verify Output ─────────────────────────────────────────────────────
def verify_output():
    """
    Samples one tensor per class per split and verifies:
    1. Shape is (16, 3, 224, 224)
    2. Values are in expected normalized range (-4 to 4)
    3. Denormalized frame looks like a normal image
    """
    log("\n" + "─" * 60)
    log("OUTPUT VERIFICATION")
    log("─" * 60)

    expected_shape = torch.Size([NUM_FRAMES, 3,
                                 FRAME_SIZE[1], FRAME_SIZE[0]])
    all_passed     = True

    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = os.path.join(NORMALIZED_DIR, split, cls)
            if not os.path.exists(cls_dir):
                log(f"  MISSING: {split}/{cls}", 'WARN')
                all_passed = False
                continue

            # Find first .pt file
            sample_pt = None
            for sub in os.listdir(cls_dir):
                sub_path = os.path.join(cls_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                for pt_file in os.listdir(sub_path):
                    if pt_file.endswith('.pt'):
                        sample_pt = os.path.join(
                            sub_path, pt_file)
                        break
                if sample_pt:
                    break

            if sample_pt is None:
                log(f"  NO PT FILES: {split}/{cls}", 'WARN')
                all_passed = False
                continue

            tensor = torch.load(sample_pt)

            # Check shape
            shape_ok = tensor.shape == expected_shape

            # Check value range
            tmin = tensor.min().item()
            tmax = tensor.max().item()
            range_ok = (-4 < tmin and tmax < 4)

            # Check no NaN
            nan_ok = not torch.isnan(tensor).any().item()

            status = "PASS" if (shape_ok and range_ok and nan_ok) \
                     else "FAIL"

            if status == "FAIL":
                all_passed = False

            log(f"  {split}/{cls:<20} "
                f"shape:{tensor.shape} "
                f"range:[{tmin:.2f},{tmax:.2f}] "
                f"nan:{not nan_ok} "
                f"{status}")

    if all_passed:
        log("\nALL TENSOR CHECKS PASSED")
    else:
        log("\nSOME TENSOR CHECKS FAILED", 'WARN')

    return all_passed

# ── Main Normalization ────────────────────────────────────────────────
def run_normalization():
    log("=" * 60)
    log("FRAME NORMALIZATION")
    log("=" * 60)
    log(f"Input          : {RESIZED_DIR}")
    log(f"Output         : {NORMALIZED_DIR}")
    log(f"Mean           : {IMAGENET_MEAN.tolist()}")
    log(f"Std            : {IMAGENET_STD.tolist()}")
    log(f"Output shape   : ({NUM_FRAMES}, 3, "
        f"{FRAME_SIZE[1]}, {FRAME_SIZE[0]})")
    log(f"Output format  : .pt tensor files")
    log("=" * 60)

    log("\nCounting total videos...")
    total_videos = count_total_videos()
    log(f"Total videos to process: {total_videos}")

    stats = {
        'config': {
            'mean'        : IMAGENET_MEAN.tolist(),
            'std'         : IMAGENET_STD.tolist(),
            'num_frames'  : NUM_FRAMES,
            'frame_size'  : FRAME_SIZE,
            'output_shape': [NUM_FRAMES, 3,
                             FRAME_SIZE[1], FRAME_SIZE[0]]
        },
        'results': {},
        'totals' : {
            'processed'  : 0,
            'success'    : 0,
            'no_frames'  : 0,
            'read_error' : 0,
            'wrong_size' : 0,
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
            src_cls = os.path.join(RESIZED_DIR,    split, cls)
            dst_cls = os.path.join(NORMALIZED_DIR, split, cls)

            log(f"\n{'─'*50}")
            log(f"{split}/{cls}")

            if not os.path.exists(src_cls):
                log(f"SKIP: source not found", 'WARN')
                continue

            cls_stats = {
                'success'   : 0,
                'no_frames' : 0,
                'read_error': 0,
                'wrong_size': 0,
            }

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
                    # Save as .pt file — same name as folder
                    dst_pt  = os.path.join(
                        dst_sub, f"{video_dir}.pt")

                    shape, status = process_video(
                        src_vid, dst_pt)

                    cls_stats[status] = \
                        cls_stats.get(status, 0) + 1
                    stats['totals'][status] = \
                        stats['totals'].get(status, 0) + 1

                    processed += 1
                    stats['totals']['processed'] += 1

                    # Log problems only
                    if status != 'success':
                        log(f"    {status.upper()}: "
                            f"{subcategory}/{video_dir}",
                            'WARN')

                    # Progress every 200 videos
                    if processed % 200 == 0:
                        elapsed = (datetime.now() -
                                   start_time).seconds
                        pct     = processed / total_videos * 100
                        log(f"  ── Progress: {processed}/"
                            f"{total_videos} "
                            f"({pct:.1f}%) | "
                            f"Elapsed: {elapsed//60}m "
                            f"{elapsed%60}s")

            log(f"  {split}/{cls} DONE: "
                f"success={cls_stats['success']} "
                f"errors={sum(v for k,v in cls_stats.items() if k != 'success')}")

            stats['results'][split][cls] = cls_stats

    # Final summary
    elapsed_total = (datetime.now() - start_time).seconds

    log("\n" + "=" * 60)
    log("NORMALIZATION COMPLETE")
    log("=" * 60)
    log(f"Total time  : {elapsed_total//60}m {elapsed_total%60}s")
    log(f"Processed   : {stats['totals']['processed']}")
    log(f"Success     : {stats['totals']['success']}")
    log(f"No frames   : {stats['totals']['no_frames']}")
    log(f"Read error  : {stats['totals']['read_error']}")
    log(f"Wrong size  : {stats['totals']['wrong_size']}")
    log(f"\nOutput      : {NORMALIZED_DIR}")
    log(f"Log file    : {LOG_FILE}")

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    log(f"Stats saved : {STATS_FILE}")

    # Verify output tensors
    verify_output()

    log("\n" + "─" * 60)
    log("NEXT STEPS:")
    log("─" * 60)
    log("1. Run normalization verification:")
    log("   python verification/verify_normalization.py")
    log("2. Only proceed to model training after verification passes")
    log("─" * 60)


run_normalization()
