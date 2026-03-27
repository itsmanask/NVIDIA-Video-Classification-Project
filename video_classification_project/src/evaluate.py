"""
evaluate.py

Standalone test set evaluation for VideoClassifier.
Loads the best Phase 2 checkpoint and runs full inference on the test split.

Outputs:
    logs/evaluate.log              ← timestamped run log
    logs/evaluate_results.json     ← all metrics in structured form
    logs/confusion_matrix.png      ← visual confusion matrix (counts + %)
    logs/misclassified_videos.csv  ← every wrong prediction with filename

What this script measures:
    - Overall test accuracy
    - Per-class accuracy (precision per class)
    - Confusion matrix — rows = true label, cols = predicted label
    - Misclassified video list — filename, true label, predicted label,
      and confidence score of the wrong prediction

Usage:
    cd /workspace/NVIDIA-Video-Classification-Project/video_classification_project
    python src/evaluate.py

This script is fully standalone — it does not import anything from train.py.
Run it only ONCE on the test set, after all training decisions are finalised.
"""

import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Project root on sys.path so src.* imports work when called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_dataloaders import get_dataloaders, IDX_TO_CLASS, CLASS_TO_IDX
from src.model import build_model


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE_DIR        = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
NORMALIZED_DIR  = os.path.join(BASE_DIR, 'frames_normalized')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'phase2_best.pt')
LOG_DIR         = os.path.join(BASE_DIR, 'logs')
EVAL_LOG        = os.path.join(LOG_DIR,  'evaluate.log')
RESULTS_JSON    = os.path.join(LOG_DIR,  'evaluate_results.json')
CONFUSION_PNG   = os.path.join(LOG_DIR,  'confusion_matrix.png')
MISCLASSIFIED   = os.path.join(LOG_DIR,  'misclassified_videos.csv')

# Class order — must match training label mapping exactly
CLASSES     = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']
NUM_CLASSES = 4

# DataLoader settings — num_workers=0 for eval (safe, no multiprocess needed)
BATCH_SIZE  = 32
NUM_WORKERS = 0


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

def log(msg, level='INFO'):
    ts   = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(EVAL_LOG, 'a') as f:
        f.write(line + '\n')


# ══════════════════════════════════════════════════════════════════════
# INFERENCE — FULL TEST SET
# ══════════════════════════════════════════════════════════════════════

def run_inference(model, loader, device):
    """
    Runs full inference over the test DataLoader.

    Returns:
        all_preds   : list[int]   — predicted class index per video
        all_labels  : list[int]   — true class index per video
        all_confs   : list[float] — softmax confidence of predicted class
        all_paths   : list[str]   — .pt file path for each video
                                    (used to identify misclassified files)
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_confs  = []
    all_paths  = []

    # Grab the underlying dataset to access file paths
    dataset = loader.dataset

    with torch.no_grad():
        for batch_idx, (tensors, labels) in enumerate(loader):
            tensors = tensors.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits = model(tensors)                         # (B, 4)
            probs  = torch.softmax(logits, dim=1)           # (B, 4)
            preds  = probs.argmax(dim=1)                    # (B,)
            confs  = probs.max(dim=1).values                # (B,)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

            # Resolve file paths for this batch
            start = batch_idx * loader.batch_size
            end   = min(start + loader.batch_size, len(dataset))
            for i in range(start, end):
                pt_path, _ = dataset.samples[i]
                all_paths.append(pt_path)

    return all_preds, all_labels, all_confs, all_paths


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(all_preds, all_labels):
    """
    Computes overall accuracy, per-class accuracy, and confusion matrix.

    Confusion matrix layout:
        rows    = true class
        columns = predicted class
        cm[i][j] = number of videos truly class i predicted as class j
        Diagonal = correct predictions

    Returns:
        overall_acc   : float
        per_class_acc : dict {class_name: accuracy_float}
        cm            : np.ndarray shape (4, 4), raw counts
    """
    total   = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / total * 100.0

    # Confusion matrix — raw counts
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1

    # Per-class accuracy = diagonal / row total
    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        row_total = cm[i].sum()
        cls_correct = cm[i][i]
        per_class_acc[cls] = (cls_correct / row_total * 100.0
                              if row_total > 0 else 0.0)

    return overall_acc, per_class_acc, cm


# ══════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, save_path):
    """
    Saves a confusion matrix PNG with both raw counts and row percentages.

    Each cell shows:
        N        ← raw count
        (XX.X%)  ← percentage of that true class (row-normalised)
    Diagonal cells (correct) are darker green.
    Off-diagonal cells (errors) are shades of red proportional to count.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Row-normalise for colour mapping (0.0–1.0)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Plot using a diverging palette — green diagonal, red off-diagonal
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='RdYlGn',
                   vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-normalised accuracy', fontsize=10)

    # Axis labels
    tick_marks = np.arange(NUM_CLASSES)
    short_names = ['Animation', 'Flat\nContent', 'Gaming', 'Natural\nContent']

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=12, labelpad=10)
    ax.set_ylabel('True label', fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix — Test Set', fontsize=14, pad=15)

    # Cell annotations: count + percentage
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            count = cm[i][j]
            pct   = cm_norm[i][j] * 100

            # White text on dark cells, black on light cells
            text_color = 'white' if (cm_norm[i][j] > 0.6 or
                                     cm_norm[i][j] < 0.15) else 'black'

            ax.text(j, i,
                    f"{count}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color=text_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# MISCLASSIFIED CSV
# ══════════════════════════════════════════════════════════════════════

def save_misclassified(all_preds, all_labels, all_confs, all_paths, save_path):
    """
    Writes one row per misclassified video to a CSV file.

    Columns:
        filename       ← basename of the .pt file
        full_path      ← full path on disk (for debugging)
        true_label     ← correct class name
        predicted      ← what the model predicted
        confidence     ← softmax confidence of the wrong prediction (0–1)
        subcategory    ← folder name one level above the .pt file
                         (useful for spotting subcategory-level weaknesses)
    """
    rows = []
    for pred, label, conf, path in zip(
            all_preds, all_labels, all_confs, all_paths):
        if pred != label:
            filename    = os.path.basename(path)
            subcategory = os.path.basename(os.path.dirname(path))
            rows.append({
                'filename'    : filename,
                'full_path'   : path,
                'true_label'  : IDX_TO_CLASS[label],
                'predicted'   : IDX_TO_CLASS[pred],
                'confidence'  : round(conf, 4),
                'subcategory' : subcategory,
            })

    # Sort by confidence descending — most confidently wrong at top
    rows.sort(key=lambda r: r['confidence'], reverse=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'true_label', 'predicted',
            'confidence', 'subcategory', 'full_path'])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():

    os.makedirs(LOG_DIR, exist_ok=True)
    # Fresh log — evaluation is always a clean run
    open(EVAL_LOG, 'w').close()

    log("=" * 60)
    log("VIDEO CLASSIFIER — TEST SET EVALUATION")
    log("=" * 60)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device : {device}")

    # ── Checkpoint ────────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        log(f"CHECKPOINT NOT FOUND: {CHECKPOINT_PATH}", 'ERROR')
        log("Run src/train.py first to produce phase2_best.pt", 'ERROR')
        sys.exit(1)

    log(f"Checkpoint : {CHECKPOINT_PATH}")

    # ── Model ─────────────────────────────────────────────────────────
    log("\nBuilding model...")
    # freeze_backbone=False here — we load all weights immediately after,
    # so the freeze state doesn't matter. eval() is called inside run_inference.
    model, device = build_model(
        num_classes     = NUM_CLASSES,
        freeze_backbone = False,
        device          = device,
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    log(f"Weights loaded from epoch {ckpt['epoch']} "
        f"(Phase {ckpt['phase']} best val acc: {ckpt['best_val_acc']:.2f}%)")

    # ── DataLoader — test split only ──────────────────────────────────
    log("\nBuilding test dataloader...")
    # get_dataloaders builds all three splits; we only use 'test'
    dataloaders = get_dataloaders(
        NORMALIZED_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )
    test_loader = dataloaders['test']
    log(f"Test samples  : {len(test_loader.dataset)}")
    log(f"Test batches  : {len(test_loader)}")

    # ── Inference ─────────────────────────────────────────────────────
    log("\nRunning inference on test set...")
    t0 = datetime.now()

    all_preds, all_labels, all_confs, all_paths = run_inference(
        model, test_loader, device)

    elapsed = (datetime.now() - t0).seconds
    log(f"Inference complete in {elapsed}s")

    # ── Metrics ───────────────────────────────────────────────────────
    overall_acc, per_class_acc, cm = compute_metrics(all_preds, all_labels)

    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"Overall test accuracy : {overall_acc:.2f}%")
    log(f"Total videos          : {len(all_labels)}")
    log(f"Correct               : {sum(p==l for p,l in zip(all_preds, all_labels))}")
    log(f"Wrong                 : {sum(p!=l for p,l in zip(all_preds, all_labels))}")

    log("\nPer-class accuracy:")
    for cls in CLASSES:
        row_total = cm[CLASS_TO_IDX[cls]].sum()
        log(f"  {cls:<20} {per_class_acc[cls]:6.2f}%  "
            f"({cm[CLASS_TO_IDX[cls]][CLASS_TO_IDX[cls]]}/{row_total} correct)")

    # ── Confusion matrix text ──────────────────────────────────────────
    log("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"  {'':20}" + "".join(f"{c[:8]:>10}" for c in CLASSES)
    log(header)
    for i, true_cls in enumerate(CLASSES):
        row_str = f"  {true_cls:<20}" + "".join(
            f"{cm[i][j]:>10}" for j in range(NUM_CLASSES))
        log(row_str)

    # ── Confusion matrix plot ─────────────────────────────────────────
    plot_confusion_matrix(cm, CONFUSION_PNG)
    log(f"\nConfusion matrix plot saved : {CONFUSION_PNG}")

    # ── Misclassified CSV ─────────────────────────────────────────────
    n_wrong = save_misclassified(
        all_preds, all_labels, all_confs, all_paths, MISCLASSIFIED)
    log(f"Misclassified videos saved  : {MISCLASSIFIED}")
    log(f"  ({n_wrong} videos misclassified, "
        f"sorted by confidence descending)")

    # ── Per-class confusion breakdown ──────────────────────────────────
    log("\nDetailed confusion breakdown (what each class gets confused with):")
    for i, true_cls in enumerate(CLASSES):
        row_total = cm[i].sum()
        errors    = [(CLASSES[j], cm[i][j])
                     for j in range(NUM_CLASSES) if j != i and cm[i][j] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)

        if not errors:
            log(f"  {true_cls:<20} → no misclassifications")
        else:
            error_str = "  |  ".join(
                f"{cls}: {count} ({count/row_total*100:.1f}%)"
                for cls, count in errors)
            log(f"  {true_cls:<20} → {error_str}")

    # ── Save JSON results ─────────────────────────────────────────────
    results = {
        'checkpoint'      : CHECKPOINT_PATH,
        'checkpoint_epoch': ckpt['epoch'],
        'checkpoint_phase': ckpt['phase'],
        'best_val_acc'    : ckpt['best_val_acc'],
        'overall_test_acc': round(overall_acc, 4),
        'total_videos'    : len(all_labels),
        'correct'         : sum(p == l for p, l in zip(all_preds, all_labels)),
        'wrong'           : sum(p != l for p, l in zip(all_preds, all_labels)),
        'per_class_acc'   : {cls: round(acc, 4)
                             for cls, acc in per_class_acc.items()},
        'confusion_matrix': cm.tolist(),
        'class_order'     : CLASSES,
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nFull results saved          : {RESULTS_JSON}")

    log("\n" + "=" * 60)
    log("EVALUATION COMPLETE")
    log("=" * 60)
    log(f"  Overall test accuracy : {overall_acc:.2f}%")
    log(f"  Confusion matrix      : {CONFUSION_PNG}")
    log(f"  Misclassified videos  : {MISCLASSIFIED}")
    log(f"  Full results JSON     : {RESULTS_JSON}")
    log("=" * 60)


if __name__ == '__main__':
    main()
