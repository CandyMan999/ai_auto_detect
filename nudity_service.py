# nudity_service.py
import os
import cv2  # frame extraction only
import requests
import tempfile
import pathlib
import threading
from typing import List, Dict, Any, Optional

from nudenet import NudeDetector

# --------------------------- Config ---------------------------

NUDITY_TAGS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}

THRESHOLD        = float(os.getenv("NUDITY_THRESHOLD", "0.5"))
MAX_VIDEO_BYTES  = int(os.getenv("MAX_VIDEO_BYTES", str(50 * 1024 * 1024)))  # 50MB
MAX_FRAMES       = int(os.getenv("MAX_FRAMES", "9"))                         # up to 9 frames (10..90%)

UA = {"User-Agent": "nudity-service/1.0"}

# Writable path on Heroku; will re-download on dyno restart
MODEL_PATH = os.getenv("NUDENET_MODEL_PATH", "/tmp/models/nudenet.onnx")
MODEL_URL  = os.getenv("NUDENET_MODEL_URL")  # e.g., your Dropbox direct link (dl=1)

# --------------------------- Model ensure ---------------------------

def _ensure_model(override_path: Optional[str] = None) -> str:
    """
    Ensure the ONNX exists locally. If missing, download from MODEL_URL.
    Returns absolute path to the model file.
    """
    dest = pathlib.Path(override_path or MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If present and plausibly large (> 50MB), reuse it
    if dest.exists() and dest.stat().st_size > 50 * 1024 * 1024:
        return str(dest.resolve())

    if not MODEL_URL:
        raise RuntimeError("NUDENET_MODEL_URL not set and model file missing")

    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": UA["User-Agent"],
    }

    tmp_path = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 5):
        try:
            print(f"[nudity] Downloading ONNX model to {dest} (attempt {attempt}) ...")
            with requests.get(
                MODEL_URL,
                stream=True,
                timeout=300,
                headers=headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)

            size = tmp_path.stat().st_size
            if size < 50 * 1024 * 1024:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded model too small: {size} bytes")

            tmp_path.replace(dest)
            print(f"[nudity] Model saved to {dest} ({size} bytes)")
            break
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed to download ONNX model: {e}") from e
            print(f"[nudity] Download failed (attempt {attempt}), retrying...")
    return str(dest.resolve())

# --------------------------- Singleton detector ---------------------------

_DETECTOR: Optional[NudeDetector] = None
_DETECTOR_LOCK = threading.Lock()
_MODEL_FILE: Optional[str] = None

def get_detector(override_path: Optional[str] = None) -> NudeDetector:
    global _DETECTOR, _MODEL_FILE
    if _DETECTOR is None:
        with _DETECTOR_LOCK:
            if _DETECTOR is None:
                _MODEL_FILE = _ensure_model(override_path)
                # Lower inference_resolution slightly to reduce RAM
                _DETECTOR = NudeDetector(model_path=_MODEL_FILE, inference_resolution=512)
                print(f"[nudity] NudeDetector initialized once with model {_MODEL_FILE}")
    return _DETECTOR

# --------------------------- Helpers ---------------------------

def _analyze(detection_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compatible with NudeNet outputs using either 'label' or legacy 'class' key.
    """
    out = {"nudity_detected": False, "details": []}
    for frame_number, result in enumerate(detection_results or []):
        for det in result or []:
            label = det.get("label", det.get("class"))
            score = float(det.get("score", 0.0))
            if label in NUDITY_TAGS and score >= THRESHOLD:
                out["nudity_detected"] = True
                out["details"].append({
                    "frame": frame_number,
                    "type": label,
                    "confidence": score,
                })
    return out

def _download_video(url: str) -> str:
    """
    Streams the video to a temp file in /tmp and enforces a size cap.
    Returns the temp file path.
    """
    print(f"[nudity] Downloading video: {url}")
    r = requests.get(url, stream=True, timeout=300, headers=UA, allow_redirects=True)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_VIDEO_BYTES:
                f.close()
                try: os.unlink(f.name)
                except Exception: pass
                raise RuntimeError(f"Video too large (> {MAX_VIDEO_BYTES // (1024*1024)} MB)")
            f.write(chunk)
        print(f"[nudity] Video saved to {f.name} ({total} bytes)")
        return f.name

def _extract_frames(video_path: str) -> List[str]:
    """
    Extract up to MAX_FRAMES frames at ~10..90% of the video.
    Returns list of image file paths under /tmp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError("No frames in video")

    indices = [int(total * i / 10) for i in range(1, 10)]  # 10%, 20%, ..., 90%
    frames_dir = tempfile.mkdtemp(dir="/tmp", prefix="frames_")
    paths: List[str] = []

    try:
        for i, idx in enumerate(indices):
            if len(paths) >= MAX_FRAMES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            p = os.path.join(frames_dir, f"frame_{i}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    finally:
        cap.release()

    if not paths:
        raise RuntimeError("Failed to extract frames")

    print(f"[nudity] Extracted {len(paths)} frames to {frames_dir}")
    return paths

def _cleanup_files(files: List[str], extra_paths: Optional[List[str]] = None):
    for p in files or []:
        try: os.remove(p)
        except Exception: pass
    # remove parent dirs if empty
    for d in {pathlib.Path(p).parent for p in files or []}:
        try: os.rmdir(d)
        except Exception: pass
    for p in extra_paths or []:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

# --------------------------- Public API ---------------------------

def process_video(video_url: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Download video -> sample frames -> run NudeDetector(ONNX) -> analyze -> cleanup.
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid_path: Optional[str] = None
    frame_paths: List[str] = []

    try:
        # 1) Get singleton detector (ensures model once)
        detector = get_detector(model_path)

        # 2) Download video & 3) Extract frames
        vid_path = _download_video(video_url)
        frame_paths = _extract_frames(vid_path)

        # 4) Run detection (batch) with NudeNet ONNX
        results = detector.detect_batch(frame_paths)

        # 5) Summarize
        summary = _analyze(results)

        return {
            "status": 200,
            "nudity_detected": summary["nudity_detected"],
            "details": summary["details"],
            "frames_analyzed": len(frame_paths),
        }

    except requests.HTTPError as he:
        return {"error": f"Download failed: {he}", "status": 400}
    except RuntimeError as re:
        msg = str(re)
        code = 413 if "too large" in msg.lower() else 400
        return {"error": msg, "status": code}
    except Exception as e:
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        _cleanup_files(frame_paths, extra_paths=[vid_path])

def nudity_health() -> Dict[str, Any]:
    """
    Health/status for the nudity subsystem.
    """
    try:
        det = get_detector()
        return {"ok": True, "model": _MODEL_FILE or MODEL_PATH, "resolution": getattr(det, "inference_resolution", "unknown")}
    except Exception as e:
        return {"ok": False, "error": str(e)}
