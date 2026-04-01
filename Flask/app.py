"""
app.py  —  Flask inference server for NVIDIA Video Classification
=================================================================
  • GPU (CUDA) auto-detection — falls back to CPU gracefully
  • Fast frame extraction with keyframe-proximity seeking
  • Terminal progress bar via tqdm with ETA
  • UI progress via Server-Sent Events (SSE)
  • Non-blocking: endpoints return job_id immediately
  • Model analysis: per-frame attention weights, visual metrics, thumbnails
  • yt-dlp URL download with real-time progress
  • Downloaded videos stored in <repo_root>/downloaded_videos

Mode 1: Test Dataset  → POST /api/classify         { selected_video }
Mode 2: Upload Video  → POST /api/classify_external { video file }
Mode 3: URL Download  → POST /api/download_url      { url }
Progress stream       → GET  /api/progress/<job_id>

Usage:
  pip install flask werkzeug torch torchvision opencv-python numpy tqdm yt-dlp
  python app.py  →  http://localhost:5000
"""

import sys, uuid, logging, threading, time, json, base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from tqdm import tqdm
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).resolve().parents[1] / 'video_classification_project'
FORCE_CPU       = False

RAW_TEST_DIR    = BASE_DIR / 'data' / 'raw' / 'test'
CHECKPOINT_PATH = BASE_DIR / 'checkpoints' / 'phase2_best.pt'
STATIC_DIR      = Path(__file__).parent / 'static'
DOWNLOAD_DIR    = Path(__file__).resolve().parents[1] / 'downloaded_videos'   # saved at base_dir/downloaded_videos
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# JOB REGISTRY
# ══════════════════════════════════════════════════════════════════════

class Job:
    __slots__ = ('id', 'status', 'progress', 'step_name', 'eta_sec',
                 'result', 'error', 'started_at', '_lock')

    def __init__(self, job_id: str):
        self.id         = job_id
        self.status     = 'queued'
        self.progress   = 0
        self.step_name  = 'Waiting…'
        self.eta_sec    = None
        self.result     = None
        self.error      = None
        self.started_at = time.time()
        self._lock      = threading.Lock()

    def update(self, *, progress=None, step_name=None, eta_sec=None, status=None):
        with self._lock:
            if progress  is not None: self.progress  = progress
            if step_name is not None: self.step_name = step_name
            if eta_sec   is not None: self.eta_sec   = eta_sec
            if status    is not None: self.status    = status

    def to_dict(self):
        with self._lock:
            return {
                'id'       : self.id,
                'status'   : self.status,
                'progress' : self.progress,
                'step_name': self.step_name,
                'eta_sec'  : self.eta_sec,
                'result'   : self.result,
                'error'    : self.error,
            }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

def new_job() -> Job:
    job_id = uuid.uuid4().hex
    job = Job(job_id)
    with _jobs_lock:
        _jobs[job_id] = job
    return job

def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


# ══════════════════════════════════════════════════════════════════════
# MODEL
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

    def forward_with_attention(self, x):
        B, T, C, H, W = x.shape
        x    = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        feat = feat.reshape(B, T, -1)
        attn = torch.softmax(self.attention(feat), dim=1)   # (B, T, 1)
        x    = (feat * attn).sum(dim=1)
        return self.classifier(x), attn.squeeze(-1)         # logits, (B, T)


_model      = None
_device     = None
_model_lock = threading.Lock()


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    with _model_lock:
        if _model is not None:
            return _model, _device
        _device = (torch.device('cuda') if not FORCE_CPU and torch.cuda.is_available()
                   else torch.device('cpu'))
        log.info(f"Device: {_device}" + (f" — {torch.cuda.get_device_name(0)}"
                                          if _device.type == 'cuda' else ""))
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}\nRun src/train.py first.")
        _model = VideoClassifier(NUM_CLASSES).to(_device)
        ckpt   = torch.load(str(CHECKPOINT_PATH), map_location=_device)
        _model.load_state_dict(ckpt['model_state_dict'])
        _model.eval()
        log.info(f"  ✓ Model loaded — epoch {ckpt['epoch']}  "
                 f"best val acc {ckpt['best_val_acc']:.2f}%")
    return _model, _device


# ══════════════════════════════════════════════════════════════════════
# FRAME PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def _preprocess_batch_gpu(frames: list, device: torch.device) -> torch.Tensor:
    rgb_list  = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    batch_np  = np.stack(rgb_list, axis=0)
    batch_cpu = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    batch     = batch_cpu.to(device, non_blocking=True).float().div_(255.0)
    batch     = TF.resize(batch, [FRAME_SIZE[1], FRAME_SIZE[0]],
                          interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def is_black(frame: np.ndarray) -> bool:
    return frame.mean() < BLACK_THRESH


# ══════════════════════════════════════════════════════════════════════
# VISUAL METRICS
# ══════════════════════════════════════════════════════════════════════

def extract_visual_metrics(frames_bgr: list) -> dict:
    rgb_means, saturations, edge_densities = [], [], []
    for frame in frames_bgr:
        f   = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_means.append([round(float(rgb[:,:,c].mean()), 2) for c in range(3)])
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturations.append(round(float(hsv[:,:,1].mean()) / 255.0, 4))
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edge_densities.append(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2))

    return {
        'per_frame': {
            'rgb_means'     : rgb_means,
            'saturations'   : saturations,
            'edge_densities': edge_densities,
        },
        'aggregate': {
            'mean_r'           : round(float(np.mean([m[0] for m in rgb_means])), 2),
            'mean_g'           : round(float(np.mean([m[1] for m in rgb_means])), 2),
            'mean_b'           : round(float(np.mean([m[2] for m in rgb_means])), 2),
            'mean_saturation'  : round(float(np.mean(saturations)), 4),
            'mean_edge_density': round(float(np.mean(edge_densities)), 2),
        },
    }


def compute_attention_entropy(weights: list[float]) -> float:
    w = np.clip(np.array(weights, dtype=np.float64), 1e-9, None)
    w /= w.sum()
    return round(float(-np.sum(w * np.log(w))), 4)


def encode_frames_b64(frames_bgr: list, max_dim: int = 160) -> list[str]:
    out = []
    for frame in frames_bgr:
        h, w  = frame.shape[:2]
        scale = max_dim / max(h, w)
        thumb = (cv2.resize(frame, (int(w*scale), int(h*scale)),
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else frame)
        _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        out.append(base64.b64encode(buf).decode('utf-8'))
    return out


# ══════════════════════════════════════════════════════════════════════
# EXTRACTION + PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def extract_and_preprocess(video_path: str, job=None):
    t0 = time.time()

    def _upd(pct, name, eta=None):
        if job:
            job.update(progress=int(pct), step_name=name,
                       eta_sec=round(eta, 1) if eta is not None else None)

    def _eta(elapsed, frac):
        if frac < 0.02: return None
        return max(0.0, elapsed / frac - elapsed)

    _upd(2, 'Probing video…')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    if total < NUM_FRAMES:
        raise ValueError(f"Video too short ({total} frames < {NUM_FRAMES} required).")

    log.info(f"  Video: {total} frames  {total/fps:.1f}s  fps={fps:.1f}  "
             f"path={Path(video_path).name}")

    _upd(5, 'Building frame indices…')
    start   = int(SKIP_EDGES * total)
    end     = int((1 - SKIP_EDGES) * total)
    indices = np.unique(np.linspace(start, end - 1, N_CANDIDATES, dtype=int)).tolist()
    n_idx   = len(indices)

    _upd(10, 'Extracting frames…')
    t_read = time.time()
    pbar   = tqdm(total=n_idx, desc=f'  Frames ({Path(video_path).stem[:30]})',
                  unit='fr', leave=False,
                  bar_format='{l_bar}{bar:28} {n}/{total} ETA {remaining}',
                  dynamic_ncols=True)

    LOOKBACK   = 30
    cap2       = cv2.VideoCapture(str(video_path))
    raw_frames = []
    milestones = {int(n_idx * q) for q in (0.25, 0.50, 0.75, 1.0)}

    for i, target_idx in enumerate(indices):
        seek_to = max(0, target_idx - LOOKBACK)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        pos = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
        frame_out = None
        while pos <= target_idx:
            ret, frame = cap2.read()
            if not ret: break
            if pos == target_idx:
                frame_out = frame
                break
            pos += 1
        if frame_out is not None:
            raw_frames.append(frame_out)
        pbar.update(1)
        if i in milestones:
            elapsed = time.time() - t_read
            frac    = (i + 1) / n_idx
            _upd(int(10 + frac * 65), 'Extracting frames…', _eta(elapsed, frac))

    cap2.release()
    pbar.close()
    log.info(f"  Read {len(raw_frames)}/{n_idx} frames in {time.time()-t_read:.2f}s")

    if not raw_frames:
        raise ValueError("Failed to read any frames from video.")

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

    _upd(78, 'Preprocessing on GPU…')
    t_pp = time.time()
    _, device = get_model()
    batch = _preprocess_batch_gpu(selected, device)
    if batch.shape[0] < NUM_FRAMES:
        pad   = batch[-1:].expand(NUM_FRAMES - batch.shape[0], -1, -1, -1)
        batch = torch.cat([batch, pad], dim=0)
    tensor = batch.unsqueeze(0)

    log.info(f"  GPU preprocess: {time.time()-t_pp:.3f}s")
    _upd(90, 'Building tensor…', _eta(time.time() - t0, 0.90))

    return tensor, status, len(selected), selected


# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════

def run_inference(tensor: torch.Tensor, model, device) -> dict:
    tensor = tensor.to(device, non_blocking=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits, attn_weights = model.forward_with_attention(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0]

    probs_list = probs.cpu().tolist()
    pred_idx   = int(torch.argmax(probs).item())
    return {
        'category'         : CLASSES[pred_idx],
        'confidence'       : round(probs_list[pred_idx] * 100.0, 2),
        'all_scores'       : {cls: round(probs_list[i] * 100.0, 2)
                              for i, cls in enumerate(CLASSES)},
        'attention_weights': [round(w, 6) for w in attn_weights[0].cpu().tolist()],
    }


# ══════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def _run_pipeline(job: Job, video_path: str, source: str,
                  video_name: str, true_class: str | None):
    try:
        job.update(status='running', progress=1, step_name='Starting…')
        t0 = time.time()

        pbar_overall = tqdm(total=100, desc=f'  {video_name[:45]}', unit='%',
                            bar_format='  {l_bar}{bar:32}{r_bar}', dynamic_ncols=True)
        _prev_pct = [0]

        def _tick():
            while job.status == 'running':
                p = job.progress
                if p != _prev_pct[0]:
                    pbar_overall.update(p - _prev_pct[0])
                    _prev_pct[0] = p
                time.sleep(0.15)
        threading.Thread(target=_tick, daemon=True).start()

        tensor, status, n_frames, selected_frames = extract_and_preprocess(video_path, job)

        elapsed = time.time() - t0
        job.update(progress=93, step_name='Running model inference…',
                   eta_sec=round(max(0, elapsed * (7/93)), 1))

        model, device = get_model()
        result        = run_inference(tensor, model, device)
        t_total       = time.time() - t0

        job.update(progress=95, step_name='Extracting visual metrics…')
        visual_metrics = extract_visual_metrics(selected_frames)
        attn_entropy   = compute_attention_entropy(result['attention_weights'])

        job.update(progress=97, step_name='Encoding frame thumbnails…')
        frame_thumbnails = encode_frames_b64(selected_frames, max_dim=160)

        result.update({
            'video_name'       : video_name,
            'source'           : source,
            'frames_extracted' : n_frames,
            'extraction_status': status,
            'processing_time_s': round(t_total, 2),
            'device'           : str(device),
            'attention_entropy': attn_entropy,
            'visual_metrics'   : visual_metrics,
            'frame_thumbnails' : frame_thumbnails,
        })
        if true_class:
            result['true_class'] = true_class

        job.update(progress=100, step_name='Done', status='done', eta_sec=0)
        job.result = result

        if _prev_pct[0] < 100:
            pbar_overall.update(100 - _prev_pct[0])
        pbar_overall.close()

        log.info(f"  ✓ {video_name}  →  {result['category']}  "
                 f"({result['confidence']:.1f}%)  in {t_total:.2f}s  device={device}")

    except Exception as exc:
        log.exception(f"Pipeline error for {video_name}")
        job.update(status='error', step_name='Error')
        job.error = str(exc)


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
        if not cls_dir.exists(): continue
        entries = []
        for sub in sorted(cls_dir.iterdir()):
            if not sub.is_dir(): continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in VIDEO_EXTS:
                    entries.append(f"{cls}/{sub.name}/{f.stem}")
        if entries:
            result[cls] = entries
    return result


def resolve_video_path(video_key: str):
    parts = video_key.split('/', 2)
    if len(parts) != 3: return None, None, None
    cls, sub, stem = parts
    sub_dir = RAW_TEST_DIR / cls / sub
    if not sub_dir.exists(): return None, None, None
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
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='pipeline')


@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')

@app.route('/nvidia-logo-vert-wht.png')
def nvidia_logo():
    return send_from_directory(str(STATIC_DIR), 'nvidia-logo-vert-wht.png')

@app.route('/api/status')
def api_status():
    model_ok   = CHECKPOINT_PATH.exists()
    device_str = (f'cuda ({torch.cuda.get_device_name(0)})'
                  if not FORCE_CPU and torch.cuda.is_available() else 'cpu')
    return jsonify({
        'model_loaded'                 : model_ok,
        'data_available'               : RAW_TEST_DIR.exists(),
        'external_processing_available': model_ok,
        'ytdlp_available'              : YTDLP_AVAILABLE,
        'device'                       : device_str,
    })

@app.route('/api/videos')
def api_videos():
    try:
        videos = scan_test_videos()
        if not videos:
            return jsonify({'success': False,
                            'error': f"No test videos found under {RAW_TEST_DIR}."})
        return jsonify({'success': True, 'videos': videos})
    except Exception as e:
        log.exception("scan_test_videos error")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify():
    try:
        video_key = request.form.get('selected_video', '').strip()
        if not video_key:
            return jsonify({'success': False, 'error': 'No video selected.'}), 400
        video_path, true_class, display_name = resolve_video_path(video_key)
        if video_path is None:
            return jsonify({'success': False,
                            'error': f'Raw video not found for "{video_key}".'}), 404
        job = new_job()
        log.info(f"Job {job.id[:8]}  dataset → {video_key}")
        _executor.submit(_run_pipeline, job, video_path, 'dataset', display_name, true_class)
        return jsonify({'success': True, 'job_id': job.id})
    except Exception as e:
        log.exception("api_classify error")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/classify_external', methods=['POST'])
def api_classify_external():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'}), 400
        f = request.files['video']
        if not f.filename:
            return jsonify({'success': False, 'error': 'Empty filename.'}), 400
        safe_name = secure_filename(f.filename)
        save_path = DOWNLOAD_DIR / f'{uuid.uuid4().hex}_{safe_name}'
        f.save(str(save_path))
        log.info(f"Upload saved: {save_path.name}  ({save_path.stat().st_size/(1024*1024):.1f} MB)")
        job = new_job()
        log.info(f"Job {job.id[:8]}  upload → {safe_name}")
        _executor.submit(_run_pipeline, job, str(save_path), 'external', safe_name, None)
        return jsonify({'success': True, 'job_id': job.id})
    except Exception as e:
        log.exception("api_classify_external error")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download_url', methods=['POST'])
def api_download_url():
    """Start a yt-dlp download; progress streams via SSE, pipeline starts after download."""
    if not YTDLP_AVAILABLE:
        return jsonify({'success': False, 'error': 'yt-dlp not installed on this server.'}), 400
    try:
        data = request.get_json(silent=True) or {}
        url  = (data.get('url') or '').strip()
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided.'}), 400

        job = new_job()
        job.update(status='running', progress=0, step_name='Connecting…')

        def _download_and_run():
            try:
                out_stem = DOWNLOAD_DIR / uuid.uuid4().hex

                def _hook(d):
                    if d.get('status') == 'downloading':
                        downloaded = d.get('downloaded_bytes') or 0
                        total      = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                        speed      = d.get('speed') or 0
                        eta        = d.get('eta')
                        pct        = int(downloaded / total * 28) + 1 if total > 0 else 5
                        spd_str    = f'{speed/1048576:.1f} MB/s' if speed and speed > 0 else ''
                        job.update(progress=pct,
                                   step_name=f'Downloading… {spd_str}'.strip(),
                                   eta_sec=round(eta, 1) if eta is not None else None)
                    elif d.get('status') == 'finished':
                        # Signal download_complete — frontend shows toast before pipeline starts
                        job.update(progress=29, step_name='download_complete', eta_sec=0)

                ydl_opts = {
                    'outtmpl'            : str(out_stem),
                    'format'             : ('bestvideo[ext=mp4][height<=720]+'
                                            'bestaudio[ext=m4a]/'
                                            'best[ext=mp4][height<=720]/best'),
                    'merge_output_format': 'mp4',
                    'quiet'              : True,
                    'no_warnings'        : True,
                    'progress_hooks'     : [_hook],
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info       = ydl.extract_info(url, download=True)
                    video_name = (info.get('title') or 'downloaded_video')[:80]

                actual = Path(str(out_stem) + '.mp4')
                if not actual.exists():
                    candidates = list(DOWNLOAD_DIR.glob(f'{out_stem.name}*'))
                    if not candidates:
                        job.update(status='error', step_name='Error')
                        job.error = 'Download produced no file.'
                        return
                    actual = candidates[0]

                log.info(f"yt-dlp saved: {actual.name}  "
                         f"({actual.stat().st_size/(1024*1024):.1f} MB)  title={video_name}")

                # Brief pause so frontend can display the "Download complete" toast
                time.sleep(1.5)

                log.info(f"Job {job.id[:8]}  url → {video_name}")
                _run_pipeline(job, str(actual), 'url', video_name, None)

            except Exception as exc:
                log.exception(f"Download error for job {job.id[:8]}")
                job.update(status='error', step_name='Error')
                job.error = str(exc)

        _executor.submit(_download_and_run)
        return jsonify({'success': True, 'job_id': job.id})

    except Exception as e:
        log.exception("api_download_url error")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/progress/<job_id>')
def api_progress(job_id: str):
    job = get_job(job_id)
    if job is None:
        def _not_found():
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return Response(stream_with_context(_not_found()), mimetype='text/event-stream')

    def _stream():
        last_sent = {}
        while True:
            d = job.to_dict()
            if d != last_sent:
                yield f"data: {json.dumps(d)}\n\n"
                last_sent = d.copy()
            if d['status'] in ('done', 'error'):
                break
            time.sleep(0.3)

    return Response(stream_with_context(_stream()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


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
    log.info(f"  Device          : {_device}")
    log.info(f"  Checkpoint      : {CHECKPOINT_PATH}")
    log.info(f"  Test data       : {RAW_TEST_DIR}")
    log.info(f"  Downloaded vids : {DOWNLOAD_DIR}")
    log.info(f"  yt-dlp          : {'available' if YTDLP_AVAILABLE else 'not installed'}")
    log.info("  URL             : http://localhost:5000")
    log.info("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
