# nudity_service.py
import os
import cv2
import requests
import tempfile
import pathlib
from typing import List, Dict, Any, Optional

# NudeNet – will use the provided ONNX path if we pass model_path
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
UA               = {"User-Agent": "nudity-service/1.0"}

# Model path and (optional) download URL
DEFAULT_MODEL_PATH = os.getenv("NUDENET_MODEL_PATH", "/tmp/models/nudenet.onnx")
MODEL_URL          = os.getenv("https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx")  # e.g. https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx


# --------------------------- Helpers ---------------------------

def _ensure_model(model_path: Optional[str]) -> str:
    """
    Ensure an ONNX model is present. If `model_path` is given, prefer it.
    Otherwise use DEFAULT_MODEL_PATH. If the target file doesn't exist and
    MODEL_URL is set, download it. Returns an absolute path to the model.
    """
    path = pathlib.Path(model_path or DEFAULT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    # If exists and looks sane (>1MB), use it
    if path.exists() and path.stat().st_size > 1 * 1024 * 1024:
        print(f"[nudity] Using existing model at: {path}")
        return str(path.resolve())

    # Download if URL provided
    if not MODEL_URL:
        raise RuntimeError(
            f"ONNX model missing at {path} and NUDENET_MODEL_URL is not set"
        )

    print(f"[nudity] Downloading ONNX model from {MODEL_URL} to {path} ...")
    for attempt in range(1, 5):
        try:
            with requests.get(MODEL_URL, stream=True, timeout=300, headers=UA, allow_redirects=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            if path.stat().st_size < 5 * 1024 * 1024:
                raise RuntimeError(f"Downloaded model too small: {path.stat().st_size} bytes")
            print("[nudity] Model downloaded successfully.")
            return str(path.resolve())
        except Exception as e:
            # Clean partial file
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            if attempt == 4:
                raise RuntimeError(f"Failed to download ONNX model: {e}")
            print(f"[nudity] Download failed (attempt {attempt}), retrying...")

    # Should never reach here
    return str(path.resolve())


def _analyze(detection_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = {"nudity_detected": False, "details": []}
    for frame_number, result in enumerate(detection_results or []):
        for det in result or []:
            if det.get("class") in NUDITY_TAGS and float(det.get("score", 0.0)) >= THRESHOLD:
                out["nudity_detected"] = True
                out["details"].append({
                    "frame": frame_number,
                    "type": det.get("class"),
                    "confidence": float(det.get("score", 0.0)),
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
        try:
            os.remove(p)
        except Exception:
            pass
    # remove parent dirs if empty
    for d in {pathlib.Path(p).parent for p in files or []}:
        try:
            os.rmdir(d)
        except Exception:
            pass
    for p in extra_paths or []:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# --------------------------- Main entry ---------------------------

def process_video(video_url: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Download video -> sample frames -> run NudeDetector(ONNX) -> analyze -> cleanup.

    Returns on success:
      {
        "status": 200,
        "nudity_detected": bool,
        "details": [{frame, type, confidence}, ...],
        "frames_analyzed": int
      }

    Returns on error (no exceptions leak back to FastAPI):
      {"error": "...", "status": <code>}
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid_path: Optional[str] = None
    frame_paths: List[str] = []

    try:
        # 1) Ensure/model resolve
        model_file = _ensure_model(model_path)
        print(f"[nudity] Using model: {model_file}")

        # 2) Download video & 3) Extract frames
        vid_path = _download_video(video_url)
        frame_paths = _extract_frames(vid_path)

        # 4) Run detection (batch)
        detector = NudeDetector(model_path=model_file, inference_resolution=640)
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
        # Catch-all to avoid 500s leaking tracebacks
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        # 6) Cleanup temp files
        _cleanup_files(frame_paths, extra_paths=[vid_path])
