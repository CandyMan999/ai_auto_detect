# nudity_service.py
import os
import cv2
import requests
import tempfile
import pathlib
from typing import List, Dict, Any, Optional

from nudenet import NudeDetector  # uses provided ONNX if model_path is passed

# Detection settings
NUDITY_TAGS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}
THRESHOLD       = float(os.getenv("NUDITY_THRESHOLD", "0.5"))
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(50 * 1024 * 1024)))  # 50MB cap
MAX_FRAMES      = int(os.getenv("MAX_FRAMES", "9"))                         # up to 9 frames (10..90%)
UA = {"User-Agent": "nudity-service/1.0"}

# Model placement (ONNX)
# Absolute path to repo-root/models/640m.onnx (works locally & on Heroku)
MODEL_PATH = str((pathlib.Path(__file__).parent / "models" / "640m.onnx").resolve())
MODEL_URL = None  # we’re not downloading anything

def _ensure_model() -> str:
    p = pathlib.Path(MODEL_PATH)
    if p.exists() and p.stat().st_size > 1 * 1024 * 1024:
        return str(p.resolve())
    raise RuntimeError(f"ONNX model not found at {p}")


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
    r = requests.get(url, stream=True, timeout=300, headers=UA, allow_redirects=True)
    r.raise_for_status()
    # Heroku-safe temp file in /tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_VIDEO_BYTES:
                f.close()
                try:
                    os.unlink(f.name)
                except Exception:
                    pass
                raise RuntimeError(f"Video too large (> {MAX_VIDEO_BYTES // (1024*1024)} MB)")
            f.write(chunk)
        return f.name

def _extract_frames(video_path: str) -> List[str]:
    """
    Extract up to MAX_FRAMES frames at ~10..90% of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError("No frames in video")

    # target frame indices ~10%, 20%, ..., 90%
    indices = [int(total * i / 10) for i in range(1, 10)]
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

    return paths

def _cleanup_files(files: List[str]):
    for p in files:
        try:
            os.remove(p)
        except Exception:
            pass
    # remove parent dirs if empty
    for d in {pathlib.Path(p).parent for p in files}:
        try:
            os.rmdir(d)
        except Exception:
            pass

def process_video(video_url: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Download video -> sample frames -> run NudeDetector(ONNX) -> analyze -> cleanup.
    Returns:
      {
        "status": 200,
        "nudity_detected": bool,
        "details": [{frame, type, confidence}, ...],
        "frames_analyzed": int
      }
    On error:
      {"error": "...", "status": <code>}
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid = None
    frames: List[str] = []
    try:
        # Ensure ONNX model available (or use explicit path)
        model_file = model_path if model_path else _ensure_model()
        detector = NudeDetector(model_path=model_file, inference_resolution=640)

        # Download video & extract frames
        vid = _download_video(video_url)
        frames = _extract_frames(vid)

        # Run detection (batch)
        results = detector.detect_batch(frames)
        summary = _analyze(results)

        return {
            "status": 200,
            "nudity_detected": summary["nudity_detected"],
            "details": summary["details"],
            "frames_analyzed": len(frames),
        }
    except requests.HTTPError as he:
        return {"error": f"Download failed: {he}", "status": 400}
    except RuntimeError as re:
        # Use 413 for size errors, else 400
        msg = str(re)
        code = 413 if "too large" in msg.lower() else 400
        return {"error": msg, "status": code}
    except Exception as e:
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        # Cleanup video + frames
        if vid and os.path.exists(vid):
            try:
                os.remove(vid)
            except Exception:
                pass
        if frames:
            _cleanup_files(frames)
