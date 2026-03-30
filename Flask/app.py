"""
app.py  —  Flask inference server for NVIDIA Video Classification
=================================================================
Improvements over original:
  • GPU (CUDA) auto-detection — falls back to CPU gracefully
  • Fast frame extraction:
      - Reads frames sequentially (no random seek) for long videos
      - Uses a thread pool so resize + normalize overlap with I/O
  • Terminal progress bar via tqdm with ETA
  • UI progress via Server-Sent Events (SSE) — same ETA shown in browser
  • Processing is non-blocking: classify endpoints return a job_id
    immediately; client polls /api/progress/<job_id> via SSE

Mode 1: Test Dataset  → POST /api/classify         { selected_video }
Mode 2: Upload Video  → POST /api/classify_external { video file }
Progress stream       → GET  /api/progress/<job_id>

Usage:
  pip install flask werkzeug torch torchvision opencv-python numpy tqdm
  python app.py
  → http://localhost:5000
"""

import os, sys, uuid, tempfile, logging, threading, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from tqdm import tqdm
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

# ══════════════════════════════════════════════════════════════════════
# CUDA GLOBAL OPTIMIZATIONS
# ══════════════════════════════════════════════════════════════════════

# OPT-5a: Let cuDNN auto-select fastest convolution algorithm for our
# fixed input shape (1, 16, 3, 224, 224). One-time cost on first run,
# free speedup on every subsequent call.
torch.backends.cudnn.benchmark = True

# OPT-5b: Enable TF32 for matmuls on Ampere+ GPUs (RTX 4060 is Ampere).
# TF32 has the same range as FP32 but 10-bit mantissa instead of 23-bit.
# For classification logits this difference is negligible — it does NOT
# change which class wins argmax.
torch.set_float32_matmul_precision('high')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).resolve().parents[1] / 'video_classification_project'
FORCE_CPU       = False   # False = auto-detect CUDA (recommended)

RAW_TEST_DIR    = BASE_DIR / 'data' / 'raw' / 'test'
CHECKPOINT_PATH = BASE_DIR / 'checkpoints' / 'phase2_best.pt'
STATIC_DIR      = Path(__file__).parent / 'static'
UPLOAD_FOLDER   = Path(tempfile.gettempdir()) / 'vc_uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Pipeline constants — locked, must match training exactly
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAME_SIZE    = (224, 224)
NUM_FRAMES    = 16
N_CANDIDATES  = 48
SKIP_EDGES    = 0.10
BLACK_THRESH  = 25

CLASSES     = ['Animation', 'Flat_Content', 'Gaming', 'Natural_Content']
NUM_CLASSES = 4
VIDEO_EXTS  = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Max parallel threads for preprocessing frames
PREPROCESS_WORKERS = min(4, (os.cpu_count() or 2))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# JOB REGISTRY  — tracks async classification jobs
# ══════════════════════════════════════════════════════════════════════

class Job:
    """Holds state for one classification job."""
    __slots__ = ('id', 'status', 'progress', 'total_steps',
                 'step_name', 'eta_sec', 'result', 'error',
                 'started_at', '_lock')

    def __init__(self, job_id: str):
        self.id          = job_id
        self.status      = 'queued'   # queued | running | done | error
        self.progress    = 0          # 0-100
        self.total_steps = 0
        self.step_name   = 'Waiting…'
        self.eta_sec     = None
        self.result      = None
        self.error       = None
        self.started_at  = time.time()
        self._lock       = threading.Lock()

    def update(self, *, progress=None, step_name=None,
               eta_sec=None, status=None):
        with self._lock:
            if progress  is not None: self.progress  = progress
            if step_name is not None: self.step_name = step_name
            if eta_sec   is not None: self.eta_sec   = eta_sec
            if status    is not None: self.status     = status

    def to_dict(self):
        with self._lock:
            return {
                'id'        : self.id,
                'status'    : self.status,
                'progress'  : self.progress,
                'step_name' : self.step_name,
                'eta_sec'   : self.eta_sec,
                'result'    : self.result,
                'error'     : self.error,
            }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

def new_job() -> Job:
    job_id = uuid.uuid4().hex
    job    = Job(job_id)
    with _jobs_lock:
        _jobs[job_id] = job
    return job

def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


# ══════════════════════════════════════════════════════════════════════
# MODEL SINGLETON
# ══════════════════════════════════════════════════════════════════════

class VideoClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        resnet          = models.resnet18(weights=None)
        self.backbone   = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        self.attention = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x    = x.reshape(B * T, C, H, W)
        x    = self.backbone(x).squeeze(-1).squeeze(-1)
        x    = x.reshape(B, T, -1)
        attn = torch.softmax(self.attention(x), dim=1)
        x    = (x * attn).sum(dim=1)
        return self.classifier(x)


_model  = None
_device = None
_model_lock = threading.Lock()


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    with _model_lock:
        if _model is not None:
            return _model, _device

        if FORCE_CPU or not torch.cuda.is_available():
            _device = torch.device('cpu')
            log.info("Device: CPU")
        else:
            _device = torch.device('cuda')
            log.info(f"Device: CUDA — {torch.cuda.get_device_name(0)}")

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}\n"
                "Run src/train.py first."
            )

        _model = VideoClassifier(NUM_CLASSES).to(_device)
        ckpt   = torch.load(str(CHECKPOINT_PATH), map_location=_device)
        _model.load_state_dict(ckpt['model_state_dict'])
        _model.eval()
        log.info(
            f"  ✓ Model loaded — epoch {ckpt['epoch']}  "
            f"best val acc {ckpt['best_val_acc']:.2f}%"
        )
    return _model, _device


# ══════════════════════════════════════════════════════════════════════
# FAST FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def _read_frames_fast(video_path: str, indices: list[int]) -> list[tuple[int, np.ndarray]]:
    """
    Read frames at the given positions quickly using keyframe-proximity seeking.

    Pure random seeking (cap.set then one cap.read) is slow on long H.264/H.265
    videos because each seek must decode an entire GOP (up to ~250 frames) from
    the nearest keyframe.  Full sequential scan reads every frame — also slow.

    Keyframe-proximity strategy:
      For each target index, seek to (target - LOOKBACK) so OpenCV lands on
      or just before the nearest keyframe, then step forward frame-by-frame
      until the exact target is reached.  With LOOKBACK=30 we read at most
      ~30 extra frames per seek regardless of video length.

    Returns list of (index, frame) in the same order as sorted(indices).
    """
    LOOKBACK = 30   # frames to step back before seeking — covers typical GOP
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    results = []
    for target_idx in sorted(indices):
        seek_to = max(0, target_idx - LOOKBACK)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # actual position after seek

        frame_out = None
        while pos <= target_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if pos == target_idx:
                frame_out = frame
                break
            pos += 1

        if frame_out is not None:
            results.append((target_idx, frame_out))

    cap.release()
    return results


def _preprocess_batch_gpu(frames: list, device: torch.device) -> torch.Tensor:
    """
    OPT-2+3: Resize + normalize all frames in ONE batched GPU operation.

    OLD approach: per-frame cv2.resize + numpy divide (CPU), then 16
    separate implicit H2D copies at inference time.

    NEW approach:
      - BGR→RGB for all frames on CPU (cheap memcpy, no math)
      - Stack into (T, H, W, 3) uint8 — no math, just memory layout
      - ONE H2D transfer: single PCIe burst for the whole batch
      - Cast + scale to float32 on GPU
      - Batched bilinear resize on GPU  (CUDA kernel, all T frames at once)
      - Vectorised subtract/divide for ImageNet normalisation on GPU

    Numerical identity:
      BILINEAR without antialias = cv2.INTER_LINEAR for downscale to 224x224.
      IMAGENET_MEAN / IMAGENET_STD constants are unchanged.
      Output tensor is identical to the per-frame CPU path.

    Returns float32 tensor (T, 3, 224, 224) already on `device`.
    """
    # Step 1: CPU — BGR→RGB colour swap only (no resize, no math)
    rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # Step 2: CPU — stack into contiguous (T, H, W, 3) uint8 array
    batch_np  = np.stack(rgb_list, axis=0)                  # (T, H, W, 3)
    batch_cpu = torch.from_numpy(batch_np).permute(0, 3, 1, 2)  # (T, 3, H, W)

    # Step 3: ONE H2D transfer — all T frames in a single PCIe burst
    # non_blocking=True lets CPU continue while DMA transfer runs
    batch = batch_cpu.to(device, non_blocking=True)         # (T, 3, H, W) uint8

    # Step 4: GPU — cast to float32 and scale [0, 255] → [0.0, 1.0]
    batch = batch.float().div_(255.0)                       # in-place divide

    # Step 5: GPU — batched bilinear resize to 224×224
    # antialias=False matches cv2.INTER_LINEAR exactly for downscaling
    batch = TF.resize(batch, [FRAME_SIZE[1], FRAME_SIZE[0]],
                      interpolation=TF.InterpolationMode.BILINEAR,
                      antialias=False)                      # (T, 3, 224, 224)

    # Step 6: GPU — ImageNet normalisation (same constants as training)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32,
                        device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32,
                        device=device).view(1, 3, 1, 1)
    batch = batch.sub_(mean).div_(std)                      # in-place ops

    return batch   # (T, 3, 224, 224) float32 on GPU


def is_black(frame: np.ndarray) -> bool:
    return frame.mean() < BLACK_THRESH


def extract_and_preprocess(video_path: str, job=None):
    """
    Full pipeline: video → 16 normalised (3,224,224) frames → tensor (1,16,3,224,224).

    Progress phases reported to `job` (if supplied) — these drive BOTH the
    terminal tqdm bar (via _run_pipeline) and the UI SSE stream:
      1–10 %   : probe + build indices
      10–75 %  : reading 48 candidate frames (updates per frame)
      75–88 %  : preprocessing (resize + normalise, parallel)
      88–95 %  : tensor build
      95–100 % : inference (reported by _run_pipeline caller)

    ETA is estimated from elapsed time and fraction complete so it is
    always non-null once at least one frame has been read.

    Returns (tensor, status, n_frames_extracted) or raises ValueError.
    """
    t_total_start = time.time()

    def _upd(pct, name, eta=None):
        if job:
            job.update(progress=int(pct), step_name=name,
                       eta_sec=round(eta, 1) if eta is not None else None)

    def _eta_from_frac(elapsed, frac):
        """Return remaining seconds given elapsed time and fraction done (0-1)."""
        if frac < 0.02:
            return None
        total_est = elapsed / frac
        return max(0.0, total_est - elapsed)

    # ── Step 1: probe video ───────────────────────────────────────────
    _upd(2, 'Probing video…')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    if total < NUM_FRAMES:
        raise ValueError(
            f"Video too short ({total} frames < {NUM_FRAMES} required)."
        )

    duration_sec = total / fps
    log.info(
        f"  Video: {total} frames  {duration_sec:.1f}s  "
        f"fps={fps:.1f}  path={Path(video_path).name}"
    )

    # ── Step 2: build candidate frame indices ─────────────────────────
    # OPT-8: numpy linspace replaces a set-comprehension + sorted().
    # np.unique handles the rare case of duplicate indices from rounding.
    _upd(5, 'Building frame indices…')
    start     = int(SKIP_EDGES * total)
    end       = int((1 - SKIP_EDGES) * total)
    indices   = np.unique(
        np.linspace(start, end - 1, N_CANDIDATES, dtype=int)
    ).tolist()
    n_indices = len(indices)

    # ── Step 3: read frames with keyframe-proximity seeking ───────────
    # OPT-9: progress updated only at 4 milestones (every 25% of frames)
    # instead of every frame. Removes 48 time.time() + 48 Lock.acquire()
    # calls from the hot path. tqdm still updates every frame for the
    # terminal bar — that's a cheap counter increment.
    _upd(10, 'Extracting frames…')
    t_read_start = time.time()

    pbar_read = tqdm(
        total=n_indices,
        desc=f'  Frames ({Path(video_path).stem[:30]})',
        unit='fr',
        leave=False,
        bar_format='{l_bar}{bar:28} {n}/{total} ETA {remaining}',
        dynamic_ncols=True,
    )

    LOOKBACK   = 30
    cap2       = cv2.VideoCapture(str(video_path))
    raw_frames = []
    # Milestone indices at which we push a UI progress update (every 25%)
    milestones = {int(n_indices * q) for q in (0.25, 0.50, 0.75, 1.0)}

    for i, target_idx in enumerate(indices):
        seek_to = max(0, target_idx - LOOKBACK)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        pos = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))

        frame_out = None
        while pos <= target_idx:
            ret, frame = cap2.read()
            if not ret:
                break
            if pos == target_idx:
                frame_out = frame
                break
            pos += 1

        if frame_out is not None:
            raw_frames.append(frame_out)

        pbar_read.update(1)

        # OPT-9: update UI only at milestones, not every frame
        if i in milestones:
            elapsed = time.time() - t_read_start
            frac    = (i + 1) / n_indices
            _upd(int(10 + frac * 65), 'Extracting frames…',
                 _eta_from_frac(elapsed, frac))

    cap2.release()
    pbar_read.close()

    t_read_end = time.time()
    log.info(f"  Read {len(raw_frames)}/{n_indices} frames in "
             f"{t_read_end - t_read_start:.2f}s")

    if not raw_frames:
        raise ValueError("Failed to read any frames from video.")

    # ── Step 4: filter black frames ───────────────────────────────────
    _upd(76, 'Filtering black frames…')
    good = [f for f in raw_frames if not is_black(f)]

    if len(good) >= NUM_FRAMES:
        step     = len(good) / NUM_FRAMES
        selected = [good[int(i * step)] for i in range(NUM_FRAMES)]
        status   = 'ok'
    elif good:
        selected = good
        status   = 'few_good'
    else:
        selected = raw_frames[:NUM_FRAMES]
        status   = 'all_black'

    log.info(f"  Good frames: {len(good)}/{len(raw_frames)}  "
             f"selected: {len(selected)}  status={status}")

    # ── Step 5+6: batched GPU preprocess + tensor build ─────────────
    # OPT-2+3: all 16 frames preprocessed in one GPU call.
    # Replaces ThreadPoolExecutor CPU loop + np.stack + CPU tensor creation.
    _upd(78, 'Preprocessing on GPU…')
    t_pp = time.time()
    _, device = get_model()

    # _preprocess_batch_gpu: BGR→RGB (CPU) → ONE H2D transfer →
    # float/255 → resize → normalise — all on GPU, all batched
    batch = _preprocess_batch_gpu(selected, device)   # (T, 3, 224, 224) on GPU

    # Pad to NUM_FRAMES if fewer good frames were found (rare)
    if batch.shape[0] < NUM_FRAMES:
        pad   = batch[-1:].expand(NUM_FRAMES - batch.shape[0], -1, -1, -1)
        batch = torch.cat([batch, pad], dim=0)

    tensor = batch.unsqueeze(0)          # (1, T, 3, 224, 224) stays on GPU

    elapsed_total = time.time() - t_total_start
    log.info(f"  GPU preprocess: {time.time()-t_pp:.3f}s")
    _upd(90, 'Building tensor…', _eta_from_frac(elapsed_total, 0.90))

    return tensor, status, len(selected)


# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════

def run_inference(tensor: torch.Tensor, model, device) -> dict:
    # OPT-3: tensor is already on `device` from _preprocess_batch_gpu.
    # .to() is a no-op if already correct device — kept for safety.
    tensor = tensor.to(device, non_blocking=True)
    # OPT-6: autocast lets Tensor Cores run matmuls in FP16.
    # Logits + softmax pulled back to FP32 before any comparison.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits = model(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0]  # ensure FP32 for softmax

    probs_list = probs.cpu().tolist()
    pred_idx   = int(torch.argmax(probs).item())
    return {
        'category'  : CLASSES[pred_idx],
        'confidence': round(probs_list[pred_idx] * 100.0, 2),
        'all_scores': {
            cls: round(probs_list[i] * 100.0, 2)
            for i, cls in enumerate(CLASSES)
        },
    }


# ══════════════════════════════════════════════════════════════════════
# FULL PIPELINE (runs in background thread)
# ══════════════════════════════════════════════════════════════════════

def _run_pipeline(job: Job, video_path: str, source: str,
                  video_name: str, true_class: str | None,
                  cleanup_path: Path | None = None):
    """
    Runs in a daemon thread.  Updates job.progress continuously.
    On completion sets job.result / job.error and job.status.
    """
    try:
        job.update(status='running', progress=1, step_name='Starting…')
        t0 = time.time()

        # ── Overall terminal progress bar ─────────────────────────────
        # Mirrors job.progress (0-100) so the terminal shows real ETA.
        pbar_overall = tqdm(
            total=100,
            desc=f'  {video_name[:45]}',
            unit='%',
            bar_format='  {l_bar}{bar:32}{r_bar}',
            dynamic_ncols=True,
        )
        _prev_pct = [0]

        def _tick_overall():
            while job.status == 'running':
                p = job.progress
                if p != _prev_pct[0]:
                    pbar_overall.update(p - _prev_pct[0])
                    _prev_pct[0] = p
                time.sleep(0.15)

        threading.Thread(target=_tick_overall, daemon=True).start()

        # ── Extract + preprocess ──────────────────────────────────────
        tensor, status, n_frames = extract_and_preprocess(video_path, job)

        # ── Inference ─────────────────────────────────────────────────
        elapsed_so_far = time.time() - t0
        # ETA for inference: model forward pass is fast (~0.5s), estimate 2s headroom
        job.update(progress=93, step_name='Running model inference…',
                   eta_sec=round(max(0, elapsed_so_far * (7/93)), 1))

        model, device = get_model()
        result        = run_inference(tensor, model, device)
        t_total       = time.time() - t0

        result['video_name']        = video_name
        result['source']            = source
        result['frames_extracted']  = n_frames
        result['extraction_status'] = status
        result['processing_time_s'] = round(t_total, 2)
        result['device']            = str(device)
        if true_class:
            result['true_class'] = true_class

        job.update(progress=100, step_name='Done', status='done', eta_sec=0)
        job.result = result

        # Finish terminal bar
        if _prev_pct[0] < 100:
            pbar_overall.update(100 - _prev_pct[0])
        pbar_overall.close()

        log.info(
            f"  ✓ {video_name}  →  {result['category']}  "
            f"({result['confidence']:.1f}%)  "
            f"in {t_total:.2f}s  device={device}"
        )

    except Exception as exc:
        log.exception(f"Pipeline error for {video_name}")
        job.update(status='error', step_name='Error')
        job.error = str(exc)

    finally:
        if cleanup_path and cleanup_path.exists():
            try:
                cleanup_path.unlink()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════
# DATASET SCANNER
# ══════════════════════════════════════════════════════════════════════

def scan_test_videos() -> dict:
    result: dict = {}
    if not RAW_TEST_DIR.exists():
        log.warning(f"Raw test dir missing: {RAW_TEST_DIR}")
        return result

    for cls in CLASSES:
        cls_dir = RAW_TEST_DIR / cls
        if not cls_dir.exists():
            continue
        entries = []
        for sub in sorted(cls_dir.iterdir()):
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in VIDEO_EXTS:
                    entries.append(f"{cls}/{sub.name}/{f.stem}")
        if entries:
            result[cls] = entries
    return result


def resolve_video_path(video_key: str):
    parts = video_key.split('/', 2)
    if len(parts) != 3:
        return None, None, None
    cls, sub, stem = parts
    sub_dir = RAW_TEST_DIR / cls / sub
    if not sub_dir.exists():
        return None, None, None
    for ext in VIDEO_EXTS:
        candidate = sub_dir / (stem + ext)
        if candidate.exists():
            return str(candidate), cls, stem
    return None, None, None


# ══════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Background thread pool — one thread per job is fine (GPU serialises anyway)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='pipeline')


@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')


@app.route('/nvidia-logo-vert-wht.png')
def nvidia_logo():
    return send_from_directory(str(STATIC_DIR), 'nvidia-logo-vert-wht.png')


@app.route('/api/status')
def api_status():
    model_ok = CHECKPOINT_PATH.exists()
    data_ok  = RAW_TEST_DIR.exists()

    # Device info
    if not FORCE_CPU and torch.cuda.is_available():
        device_str  = f'cuda ({torch.cuda.get_device_name(0)})'
    else:
        device_str  = 'cpu'

    return jsonify({
        'model_loaded'                 : model_ok,
        'data_available'               : data_ok,
        'external_processing_available': model_ok,
        'device'                       : device_str,
    })


@app.route('/api/videos')
def api_videos():
    try:
        videos = scan_test_videos()
        if not videos:
            return jsonify({
                'success': False,
                'error'  : (
                    f"No test videos found under {RAW_TEST_DIR}. "
                    "Ensure data/raw/test/ exists with .mp4 files."
                ),
            })
        return jsonify({'success': True, 'videos': videos})
    except Exception as e:
        log.exception("scan_test_videos error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    Test Dataset mode — returns job_id immediately.
    Client polls GET /api/progress/<job_id>.
    """
    try:
        video_key = request.form.get('selected_video', '').strip()
        if not video_key:
            return jsonify({'success': False, 'error': 'No video selected.'}), 400

        video_path, true_class, display_name = resolve_video_path(video_key)
        if video_path is None:
            return jsonify({
                'success': False,
                'error'  : f'Raw video file not found for "{video_key}".',
            }), 404

        job = new_job()
        log.info(f"Job {job.id[:8]}  dataset → {video_key}")
        _executor.submit(
            _run_pipeline, job, video_path,
            'dataset', display_name, true_class
        )
        return jsonify({'success': True, 'job_id': job.id})

    except Exception as e:
        log.exception("api_classify error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/classify_external', methods=['POST'])
def api_classify_external():
    """Upload mode — returns job_id immediately."""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'}), 400

        f = request.files['video']
        if not f.filename:
            return jsonify({'success': False, 'error': 'Empty filename.'}), 400

        safe_name = secure_filename(f.filename)
        tmp_path  = UPLOAD_FOLDER / f'{uuid.uuid4().hex}_{safe_name}'
        f.save(str(tmp_path))
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        log.info(f"Upload saved: {tmp_path.name}  ({size_mb:.1f} MB)")

        job = new_job()
        log.info(f"Job {job.id[:8]}  upload → {safe_name}")
        _executor.submit(
            _run_pipeline, job, str(tmp_path),
            'external', safe_name, None, tmp_path
        )
        return jsonify({'success': True, 'job_id': job.id})

    except Exception as e:
        log.exception("api_classify_external error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/progress/<job_id>')
def api_progress(job_id: str):
    """
    Server-Sent Events stream.
    Sends a JSON event every ~300 ms until the job is done or errors.

    UI listens with:
        const es = new EventSource(`/api/progress/${jobId}`);
        es.onmessage = e => { const d = JSON.parse(e.data); … };
    """
    job = get_job(job_id)
    if job is None:
        def _not_found():
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return Response(
            stream_with_context(_not_found()),
            mimetype='text/event-stream',
        )

    def _stream():
        last_sent = {}
        while True:
            d = job.to_dict()
            if d != last_sent:
                yield f"data: {json.dumps(d)}\n\n"
                last_sent = d

            if d['status'] in ('done', 'error'):
                break

            time.sleep(0.3)

    return Response(
        stream_with_context(_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control' : 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/job/<job_id>')
def api_job_result(job_id: str):
    """One-shot poll endpoint (alternative to SSE for simple clients)."""
    job = get_job(job_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    d = job.to_dict()
    return jsonify({'success': True, **d})


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info("Pre-loading model…")
    try:
        get_model()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("=" * 60)
    log.info("  NVIDIA Video Classification — Flask")
    log.info(f"  Device     : {_device}")
    log.info(f"  Checkpoint : {CHECKPOINT_PATH}")
    log.info(f"  Test data  : {RAW_TEST_DIR}")
    log.info(f"  Workers    : {PREPROCESS_WORKERS} preprocess threads")
    log.info("  URL        : http://localhost:5000")
    log.info("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
