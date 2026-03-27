"""
train.py

Two-phase training script for VideoClassifier (ResNet18 + attention pooling).

Phase 1 — Frozen backbone (20 epochs max)
    Only the classifier head + attention layer train.
    Optimizer : AdamW, lr=1e-3, weight_decay=1e-4
    Scheduler : ReduceLROnPlateau(factor=0.5, patience=2)
    Early stop: patience=5 epochs on val accuracy

Phase 2 — Fine-tune last 2 ResNet blocks (10 epochs max)
    layer3 + layer4 + classifier head + attention layer train.
    Optimizer : AdamW, lr=1e-4 (10× smaller than Phase 1)
    Scheduler : ReduceLROnPlateau(factor=0.5, patience=2)
    Early stop: patience=3 epochs on val accuracy

Key features:
    - AMP (automatic mixed precision) — uses A100 Tensor Cores
    - Gradient clipping (max_norm=1.0) — stabilises Phase 2
    - Hardware monitor thread — logs CPU/RAM/GPU every 30s
    - Checkpoint every epoch + separate best checkpoint
    - Full resume support — interrupted runs restart from last epoch

Checkpoints:
    checkpoints/phase1_latest.pt   ← overwritten every epoch (resume)
    checkpoints/phase1_best.pt     ← best val acc during Phase 1
    checkpoints/phase2_latest.pt   ← overwritten every epoch (resume)
    checkpoints/phase2_best.pt     ← best val acc during Phase 2

Logs:
    logs/train.log       ← epoch-level training stats
    logs/hardware.log    ← CPU/RAM/GPU sampled every 30s

Usage:
    cd /workspace/NVIDIA-Video-Classification-Project/video_classification_project
    python src/train.py

To resume an interrupted run, just re-run the same command.
The script auto-detects existing checkpoints and asks to resume.
"""

import os
import sys
import time
import json
import threading
from datetime import datetime

import random          # FIX 3: for reproducibility seed
import numpy as np     # FIX 3: for reproducibility seed

import torch
import torch.nn as nn
from torch.amp import autocast              # FIX 1: updated non-deprecated AMP API
from torch.cuda.amp import GradScaler       # GradScaler stays in torch.cuda.amp (torch < 2.3)

# ── psutil for CPU/RAM monitoring ─────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ── pynvml for GPU monitoring (preferred over subprocess) ─────────────
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# Project root on sys.path so `src.*` imports work when called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_dataloaders import get_dataloaders
from src.model import build_model


# ══════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY — FIX 3
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    # FIX 3: Ensure reproducible training runs across all RNG sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
# CONFIG — all tunable values live here, never inside functions
# ══════════════════════════════════════════════════════════════════════

BASE_DIR       = '/workspace/NVIDIA-Video-Classification-Project/video_classification_project'
NORMALIZED_DIR = os.path.join(BASE_DIR, 'frames_normalized')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR        = os.path.join(BASE_DIR, 'logs')
TRAIN_LOG      = os.path.join(LOG_DIR,  'train.log')
HW_LOG         = os.path.join(LOG_DIR,  'hardware.log')

# ── DataLoader ────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 8      # 8 of 10 Xeon cores — leaves 2 for OS + monitoring
PIN_MEMORY   = True   # faster CPU→GPU transfer on CUDA

# ── Phase 1 (frozen backbone) ─────────────────────────────────────────
P1_EPOCHS    = 20
P1_LR        = 1e-3
P1_PATIENCE  = 5      # early stop: epochs without val acc improvement

# ── Phase 2 (fine-tune last 2 ResNet blocks) ──────────────────────────
P2_EPOCHS    = 10
P2_LR        = 1e-4   # 10× smaller — avoids destroying pretrained weights
P2_PATIENCE  = 3      # shorter because phase is shorter

# ── Shared optimizer + scheduler settings ─────────────────────────────
WEIGHT_DECAY    = 1e-4
SCHED_FACTOR    = 0.5  # LR multiplied by this on plateau
SCHED_PATIENCE  = 2    # epochs before scheduler cuts LR

# ── Gradient clipping ─────────────────────────────────────────────────
GRAD_CLIP_NORM  = 1.0  # max gradient norm before clipping

# ── Hardware monitoring ───────────────────────────────────────────────
MONITOR_INTERVAL = 30  # seconds between hardware snapshots


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

def log(msg, level='INFO', also_print=True):
    """
    Writes a timestamped line to TRAIN_LOG.
    also_print=True mirrors output to stdout so tmux shows progress.
    """
    ts      = datetime.now().strftime('%H:%M:%S')
    line    = f"[{ts}] [{level}] {msg}"
    if also_print:
        print(line)
    with open(TRAIN_LOG, 'a') as f:
        f.write(line + '\n')


def log_hw(msg):
    """Writes a line to HW_LOG (hardware monitor output)."""
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)                      # visible in tmux
    with open(HW_LOG, 'a') as f:
        f.write(line + '\n')


# ══════════════════════════════════════════════════════════════════════
# HARDWARE MONITOR
# ══════════════════════════════════════════════════════════════════════

class HardwareMonitor:
    """
    Background thread that samples CPU, RAM, and GPU stats
    every MONITOR_INTERVAL seconds and writes to hardware.log.

    Usage:
        monitor = HardwareMonitor()
        monitor.start()
        # ... training ...
        monitor.stop()
    """

    def __init__(self, interval=MONITOR_INTERVAL):
        self.interval    = interval
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target=self._run, daemon=True, name='HWMonitor')

    def start(self):
        self._thread.start()
        log_hw("Hardware monitor started")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 2)
        log_hw("Hardware monitor stopped")

    def _run(self):
        while not self._stop_event.wait(self.interval):
            self._snapshot()

    def _snapshot(self):
        parts = []

        # ── CPU ───────────────────────────────────────────────────────
        if PSUTIL_AVAILABLE:
            cpu_pct = psutil.cpu_percent(interval=1)
            ram     = psutil.virtual_memory()
            ram_used_gb  = ram.used  / (1024 ** 3)
            ram_total_gb = ram.total / (1024 ** 3)
            parts.append(
                f"CPU {cpu_pct:5.1f}%  "
                f"RAM {ram_used_gb:.1f}/{ram_total_gb:.0f}GB "
                f"({ram.percent:.1f}%)"
            )
        else:
            parts.append("CPU/RAM: psutil not available")

        # ── GPU ───────────────────────────────────────────────────────
        if PYNVML_AVAILABLE:
            try:
                handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
                util     = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp     = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_used_gb  = mem_info.used  / (1024 ** 3)
                gpu_total_gb = mem_info.total / (1024 ** 3)
                parts.append(
                    f"GPU util {util.gpu:3d}%  "
                    f"VRAM {gpu_used_gb:.1f}/{gpu_total_gb:.1f}GB  "
                    f"Temp {temp}°C"
                )
            except Exception as e:
                parts.append(f"GPU: pynvml error ({e})")
        else:
            # Fallback: torch reports VRAM only (no utilisation %)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved  = torch.cuda.memory_reserved(0)  / (1024 ** 3)
                parts.append(
                    f"GPU VRAM allocated {allocated:.1f}GB  "
                    f"reserved {reserved:.1f}GB"
                )
            else:
                parts.append("GPU: CUDA not available")

        log_hw("  |  ".join(parts))


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(path, phase, epoch, model, optimizer,
                    scheduler, scaler, best_val_acc, best_epoch, history):
    """
    Saves full training state to disk.
    Called after every epoch (latest) and whenever best val acc improves.

    Everything needed to resume is included:
        model weights, optimizer state, scheduler state, scaler state,
        best accuracy so far, full per-epoch history list.
    """
    torch.save({
        'phase'              : phase,
        'epoch'              : epoch,
        'model_state_dict'   : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict'  : scaler.state_dict(),
        'best_val_acc'       : best_val_acc,
        'best_epoch'         : best_epoch,
        'history'            : history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """
    Restores full training state from a checkpoint file.

    Returns:
        epoch        : last completed epoch number
        best_val_acc : best val accuracy seen so far
        best_epoch   : epoch at which best_val_acc was achieved
        history      : list of per-epoch stat dicts
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return (
        ckpt['epoch'],
        ckpt['best_val_acc'],
        ckpt['best_epoch'],
        ckpt['history'],
    )


# ══════════════════════════════════════════════════════════════════════
# TRAIN / VAL — ONE EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer,
                  scaler, device, is_train):
    """
    Runs one full pass over the dataset (train or val).

    Train mode:
        - AMP autocast wraps forward + loss (TASK 2)
        - Gradient clipping before optimizer step (TASK 3)
        - scaler handles FP16 gradient scaling

    Val mode:
        - torch.no_grad() + autocast — no gradients computed
        - optimizer not touched

    Returns:
        avg_loss : float  (mean loss over all samples)
        accuracy : float  (correct / total * 100, as percentage)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for tensors, labels in loader:
        tensors = tensors.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        if is_train:
            optimizer.zero_grad()

            # TASK 2: AMP — forward + loss in float16
            with autocast(device_type='cuda'):  # FIX 1: device_type required by new API
                logits = model(tensors)
                loss   = criterion(logits, labels)

            # TASK 2: Scaled backward pass
            scaler.scale(loss).backward()

            # TASK 3: Gradient clipping — unscale first so norms are real
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            # TASK 2: Optimizer step via scaler + scaler update
            scaler.step(optimizer)
            scaler.update()

        else:
            with torch.no_grad():
                with autocast(device_type='cuda'):  # FIX 1: device_type required by new API
                    logits = model(tensors)
                    loss   = criterion(logits, labels)

        # Accumulate stats
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item() * labels.size(0)

    avg_loss = total_loss  / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# PHASE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_phase(phase_num, model, dataloaders, device,
              max_epochs, lr, patience,
              latest_ckpt_path, best_ckpt_path,
              resume_from=None):
    """
    Runs one full training phase (Phase 1 or Phase 2).

    Args:
        phase_num       : 1 or 2 (used for logging only)
        model           : VideoClassifier on device
        dataloaders     : dict with 'train' and 'val' loaders
        device          : torch.device
        max_epochs      : maximum number of epochs to run
        lr              : initial learning rate
        patience        : early stopping patience (epochs)
        latest_ckpt_path: path to save/overwrite after every epoch
        best_ckpt_path  : path to save only when val acc improves
        resume_from     : path to checkpoint to resume from (or None)

    Returns:
        history : list of dicts, one per epoch:
                  {'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'}
    """
    log("=" * 60)
    log(f"PHASE {phase_num} START")
    log(f"  max_epochs : {max_epochs}")
    log(f"  lr         : {lr}")
    log(f"  patience   : {patience}")
    log("=" * 60)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # FIX 4: lower smoothing for small 4-class dataset

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'min',        # monitor val loss
        factor   = SCHED_FACTOR,
        patience = SCHED_PATIENCE,
    )

    # TASK 2: GradScaler for AMP — handles FP16 overflow automatically
    scaler = GradScaler()

    # ── Resume state ──────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    best_epoch   = 0
    history      = []
    no_improve   = 0    # consecutive epochs without val acc improvement

    if resume_from and os.path.exists(resume_from):
        log(f"Resuming from: {resume_from}")
        start_epoch, best_val_acc, best_epoch, history = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device)
        no_improve = len(history) - best_epoch - 1
        log(f"  Resumed at epoch {start_epoch + 1}  "
            f"best_val_acc={best_val_acc:.2f}%  no_improve={no_improve}")

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs):

        epoch_start = time.time()

        # Train
        train_loss, train_acc = run_one_epoch(
            model, dataloaders['train'], criterion,
            optimizer, scaler, device, is_train=True)

        # Validate
        val_loss, val_acc = run_one_epoch(
            model, dataloaders['val'], criterion,
            optimizer, scaler, device, is_train=False)

        # Step scheduler on val loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - epoch_start

        # Log epoch summary
        log(
            f"Phase {phase_num} | "
            f"Epoch {epoch+1:02d}/{max_epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"Val loss {val_loss:.4f} acc {val_acc:.2f}% | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Record history
        epoch_stats = {
            'epoch'      : epoch + 1,
            'train_loss' : round(train_loss, 4),
            'train_acc'  : round(train_acc,  2),
            'val_loss'   : round(val_loss,   4),
            'val_acc'    : round(val_acc,    2),
            'lr'         : current_lr,
        }
        history.append(epoch_stats)

        # ── Checkpoint every epoch (latest) ───────────────────────────
        # This is the resume checkpoint — always up to date
        save_checkpoint(
            latest_ckpt_path, phase_num, epoch + 1,
            model, optimizer, scheduler, scaler,
            best_val_acc, best_epoch, history)

        # ── Save best checkpoint if val acc improved ───────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            no_improve   = 0
            save_checkpoint(
                best_ckpt_path, phase_num, epoch + 1,
                model, optimizer, scheduler, scaler,
                best_val_acc, best_epoch, history)
            log(f"  ✓ Best val acc: {best_val_acc:.2f}%  → saved {best_ckpt_path}")
        else:
            no_improve += 1
            log(f"  No improvement for {no_improve}/{patience} epochs")

        # ── Early stopping ─────────────────────────────────────────────
        if no_improve >= patience:
            log(f"Early stopping triggered at epoch {epoch+1} "
                f"(no improvement for {patience} epochs)")
            break

    log(f"\nPhase {phase_num} complete.")
    log(f"  Best val acc : {best_val_acc:.2f}%  at epoch {best_epoch}")
    log(f"  Best checkpoint : {best_ckpt_path}")
    log("=" * 60)

    return history


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():

    # ── Setup dirs ────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # Fresh log files (append mode so resume adds to existing logs)
    for path in [TRAIN_LOG, HW_LOG]:
        if not os.path.exists(path):
            open(path, 'w').close()

    log("=" * 60)
    log("VIDEO CLASSIFIER — TRAINING")
    log("=" * 60)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    if device.type == 'cuda':
        log(f"GPU   : {torch.cuda.get_device_name(0)}")
        log(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.backends.cudnn.benchmark = True  # FIX 2: enables faster convolution algorithms for fixed input sizes

    set_seed(42)  # FIX 3: ensure reproducible training runs

    # ── Hardware monitor ──────────────────────────────────────────────
    monitor = HardwareMonitor(interval=MONITOR_INTERVAL)
    monitor.start()

    # ── DataLoaders ───────────────────────────────────────────────────
    log("\nBuilding dataloaders...")
    dataloaders = get_dataloaders(
        NORMALIZED_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )
    log(f"  Train batches : {len(dataloaders['train'])}")
    log(f"  Val batches   : {len(dataloaders['val'])}")
    log(f"  Test batches  : {len(dataloaders['test'])}")

    # ── Checkpoint paths ──────────────────────────────────────────────
    P1_LATEST = os.path.join(CHECKPOINT_DIR, 'phase1_latest.pt')
    P1_BEST   = os.path.join(CHECKPOINT_DIR, 'phase1_best.pt')
    P2_LATEST = os.path.join(CHECKPOINT_DIR, 'phase2_latest.pt')
    P2_BEST   = os.path.join(CHECKPOINT_DIR, 'phase2_best.pt')

    # ── Detect existing checkpoints → decide where to start ───────────
    p2_exists = os.path.exists(P2_LATEST)
    p1_exists = os.path.exists(P1_LATEST)

    start_phase = 1
    if p2_exists:
        answer = input("\nPhase 2 checkpoint found. Resume Phase 2? [y/n]: ").strip().lower()
        if answer == 'y':
            start_phase = 2
        else:
            log("Starting fresh from Phase 1.")
            start_phase = 1
    elif p1_exists:
        answer = input("\nPhase 1 checkpoint found. Resume Phase 1? [y/n]: ").strip().lower()
        if answer == 'y':
            start_phase = 1
        else:
            log("Starting fresh from Phase 1.")

    # ── Build model ───────────────────────────────────────────────────
    log("\nBuilding model...")
    model, device = build_model(
        num_classes     = 4,
        freeze_backbone = True,   # always start frozen; Phase 2 unfreezes
        device          = device,
    )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1
    # ══════════════════════════════════════════════════════════════════

    if start_phase == 1:
        log("\n" + "─" * 60)
        log("STARTING PHASE 1 — Frozen backbone")
        log("─" * 60)

        p1_history = run_phase(
            phase_num        = 1,
            model            = model,
            dataloaders      = dataloaders,
            device           = device,
            max_epochs       = P1_EPOCHS,
            lr               = P1_LR,
            patience         = P1_PATIENCE,
            latest_ckpt_path = P1_LATEST,
            best_ckpt_path   = P1_BEST,
            resume_from      = None,   # fresh Phase 1
        )

    # ── Load best Phase 1 weights before Phase 2 ──────────────────────
    # Even if we resumed Phase 2, we only need the model weights.
    # The Phase 2 checkpoint (if resuming) will overwrite optimizer etc.
    if start_phase <= 2 and os.path.exists(P1_BEST):
        log(f"\nLoading best Phase 1 weights from: {P1_BEST}")
        ckpt = torch.load(P1_BEST, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        log("  Phase 1 weights loaded.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2
    # ══════════════════════════════════════════════════════════════════

    log("\n" + "─" * 60)
    log("STARTING PHASE 2 — Fine-tune last 2 ResNet blocks")
    log("─" * 60)

    # Unfreeze last 2 backbone blocks (layer3 + layer4)
    model.unfreeze_last_blocks(n=2)

    p2_history = run_phase(
        phase_num        = 2,
        model            = model,
        dataloaders      = dataloaders,
        device           = device,
        max_epochs       = P2_EPOCHS,
        lr               = P2_LR,
        patience         = P2_PATIENCE,
        latest_ckpt_path = P2_LATEST,
        best_ckpt_path   = P2_BEST,
        resume_from      = P2_LATEST if (start_phase == 2) else None,
    )

    # ══════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ══════════════════════════════════════════════════════════════════

    # ── FIX 5: Final test evaluation using best Phase 2 weights ──────
    # Load best Phase 2 checkpoint so we evaluate the best model, not
    # the last epoch's weights (which may have been worse).
    if os.path.exists(P2_BEST):
        ckpt = torch.load(P2_BEST, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        log("\nLoaded best Phase 2 weights for test evaluation.")

    # criterion reused for loss reporting only — model is in eval mode
    test_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    test_loss, test_acc = run_one_epoch(
        model,
        dataloaders['test'],
        test_criterion,
        optimizer = None,   # not used in is_train=False branch
        scaler    = None,   # not used in is_train=False branch
        device    = device,
        is_train  = False,
    )
    log(f"\nFinal TEST accuracy : {test_acc:.2f}%")
    log(f"Final TEST loss     : {test_loss:.4f}")

    monitor.stop()

    log("\n" + "=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    log(f"  Final model   : {P2_BEST}")
    log(f"  Train log     : {TRAIN_LOG}")
    log(f"  Hardware log  : {HW_LOG}")
    log("\nNEXT STEP:")
    log("  python src/evaluate.py")
    log("=" * 60)


if __name__ == '__main__':
    main()
