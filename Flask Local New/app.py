# app.py
import os
import tempfile
import shutil
import json
from pathlib import Path
from flask import Flask, send_file, request, jsonify
import importlib.util
import torch
import torch.nn.functional as F
import traceback

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent

# Prefer project-local folders inside BASE_DIR
FEATURES_DIR = BASE_DIR / "features_enhanced"
MODELS_DIR = BASE_DIR / "models_enhanced"
PROCESSED_TEST_DIR = BASE_DIR / "data" / "processed" / "test"

# Fallbacks if the local folders don't exist (keeps Path type)
if not FEATURES_DIR.exists():
    alt = Path("/mnt/data/features_enhanced")
    alt2 = Path("D:/features_enhanced")
    if alt.exists():
        FEATURES_DIR = alt
    elif alt2.exists():
        FEATURES_DIR = alt2

if not MODELS_DIR.exists():
    alt = Path("/mnt/data/models_enhanced")
    alt2 = Path("D:/models_enhanced")
    if alt.exists():
        MODELS_DIR = alt
    elif alt2.exists():
        MODELS_DIR = alt2

if not PROCESSED_TEST_DIR.exists():
    alt = Path("/mnt/data/data/processed/test")
    alt2 = Path("D:/data/processed/test")
    if alt.exists():
        PROCESSED_TEST_DIR = alt
    elif alt2.exists():
        PROCESSED_TEST_DIR = alt2

# Selected checkpoints to use
SELECTED_CHECKPOINTS = [
    "best_ensemble_model_1.pt",
    "best_ensemble_model_2.pt",
    "best_ensemble_model_3.pt",
    "best_ensemble_model_4.pt",
]
# ---------------- End configuration ----------------

app = Flask(__name__, static_folder=None)

# Resolve index.html and user script paths (prefer local next to app.py, fallback to /mnt/data)
INDEX_HTML_PATH = BASE_DIR / "index.html"
if not INDEX_HTML_PATH.exists():
    uploaded_index = Path("/mnt/data/index.html")
    if uploaded_index.exists():
        INDEX_HTML_PATH = uploaded_index
    else:
        INDEX_HTML_PATH = None

USER_SCRIPT_PATH = BASE_DIR / "test_already_extracted.py"
if not USER_SCRIPT_PATH.exists():
    uploaded_script = Path("/mnt/data/test_already_extracted.py")
    if uploaded_script.exists():
        USER_SCRIPT_PATH = uploaded_script
    else:
        USER_SCRIPT_PATH = None

if USER_SCRIPT_PATH is None:
    raise RuntimeError(
        "Could not find 'test_already_extracted.py' next to app.py or in /mnt/data.\n"
        "Place the script next to app.py or upload it to /mnt/data/test_already_extracted.py"
    )

# Dynamically import user's script as module "user_module"
spec = importlib.util.spec_from_file_location("user_module", str(USER_SCRIPT_PATH))
user_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_module)

# Singletons for loader & classifier (lazy)
_feature_loader = None
_classifier = None

def get_feature_loader():
    global _feature_loader
    if _feature_loader is None:
        # Use Path objects for directories (user_module expects strings or Paths)
        _feature_loader = user_module.SingleVideoFeatureLoader(
            features_dir=str(FEATURES_DIR),
            processed_test_dir=str(PROCESSED_TEST_DIR)
        )
    return _feature_loader

def get_classifier(device="cuda"):
    global _classifier
    if _classifier is None:
        # Build checkpoint paths using Path objects
        paths = []
        for name in SELECTED_CHECKPOINTS:
            p = MODELS_DIR / name
            if p.exists():
                paths.append(str(p))
            else:
                app.logger.warning(f"Checkpoint not found: {p} — skipping")
        if not paths:
            # If none found, try any .pt in MODELS_DIR
            if MODELS_DIR.exists():
                found = list(MODELS_DIR.glob("*.pt"))
                if found:
                    app.logger.warning("No selected checkpoints found, using any .pt found in models dir")
                    paths = [str(x) for x in found]
        if not paths:
            raise RuntimeError(f"No checkpoints found in {MODELS_DIR}. Expected: {SELECTED_CHECKPOINTS}")

        device_arg = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        _classifier = user_module.SingleVideoClassifier(checkpoint_paths=paths, device=device_arg)
        _classifier.load_models()
    return _classifier

# Serve index.html
@app.route("/", methods=["GET"])
def index():
    if INDEX_HTML_PATH is None:
        return ("UI file not found. Place index.html next to app.py "
                "or upload to /mnt/data/index.html"), 500
    return send_file(str(INDEX_HTML_PATH))

# API: list available dataset videos
@app.route("/api/videos", methods=["GET"])
def api_videos():
    try:
        fl = get_feature_loader()
        all_videos, videos_by_category = fl.list_available_videos()
        # Return the mapping (categories -> list)
        return jsonify({"success": True, "videos": videos_by_category})
    except Exception as e:
        app.logger.exception("Failed to list videos")
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Per-model breakdown helper
def per_model_scores(classifier, features):
    scores = []
    device = classifier.device
    features_batch = features.unsqueeze(0).to(device)  # [1, T, D]
    lengths = torch.tensor([features.shape[0]], device=device)

    with torch.no_grad():
        for i, model in enumerate(classifier.models):
            outputs = model(features_batch, lengths)  # [1, C]
            probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            # classifier.checkpoint_paths may be list of strings or Path objects
            model_name = (Path(classifier.checkpoint_paths[i]).name
                          if isinstance(classifier.checkpoint_paths[i], (str, Path))
                          else getattr(classifier.checkpoint_paths[i], 'name', str(classifier.checkpoint_paths[i])))
            scores.append({
                "model": model_name,
                "probs": probs.tolist()
            })
    return scores

# API: classify dataset video ONLY (no uploaded/external videos)
@app.route("/api/classify", methods=["POST"])
def api_classify():
    try:
        clf = get_classifier()
        fl = get_feature_loader()
    except Exception as e:
        app.logger.exception("Failed to init classifier/loader")
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

    use_tta = request.form.get("use_tta", "false").lower() == "true"
    selected_video = request.form.get("selected_video", None)

    if not selected_video:
        return jsonify({"success": False, "error": "No selected_video provided"}), 400

    try:
        # ONLY load dataset features (no URL, no uploads)
        features, label, category_name = fl.load_video_features(selected_video)
        if features is None:
            return jsonify({"success": False, "error": "Selected video not found in dataset"}), 400

        # convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)

        # Predict
        probs = clf.predict_with_tta(features) if use_tta else clf.predict_standard(features)

        final_probs = probs.numpy().tolist()
        predicted_idx = int(torch.tensor(final_probs).argmax())
        predicted_class = clf.class_names[predicted_idx]
        predicted_conf = float(final_probs[predicted_idx] * 100.0)

        # Per-model breakdown
        model_scores = per_model_scores(clf, features)
        model_scores_ui = []
        for m in model_scores:
            model_scores_ui.append({
                "model": m["model"],
                "confidence": float(max(m["probs"]) * 100.0),
                "probs": [float(p * 100.0) for p in m["probs"]]
            })

        # Build final response
        result = {
            "category": predicted_class,
            "confidence": predicted_conf,
            "model_scores": model_scores_ui,
            "all_scores": {clf.class_names[i]: float(final_probs[i] * 100.0)
                           for i in range(len(clf.class_names))},
            "video_name": selected_video
        }

        return jsonify({"success": True, "result": result})

    except Exception as e:
        app.logger.exception("Classification failed")
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500



if __name__ == "__main__":
    print("Index HTML path:", INDEX_HTML_PATH)
    print("User script path:", USER_SCRIPT_PATH)
    print("Features dir:", FEATURES_DIR)
    print("Models dir:", MODELS_DIR)
    print("Processed test dir:", PROCESSED_TEST_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)
