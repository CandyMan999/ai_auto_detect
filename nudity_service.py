# nudity_service.py
import os
import cv2  # only for frame extraction; NOT used for model loading
import requests
import tempfile
import pathlib
from typing import List, Dict, Any, Optional

# NudeNet loads the ONNX if we pass model_path
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

# Hardcoded model location + stable release URL
MODEL_PATH = "/tmp/models/640m.onnx"  # /tmp is writable on Heroku
MODEL_URL  = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx"

# --------------------------- Model ensure ---------------------------

def _ensure_model(override_path: Optional[str] = None) -> str:
    """
    Ensure the 640m.onnx exists locally. If missing, download from MODEL_URL
    with proper headers and size sanity check.
    Returns absolute path to the model file.
    """
    dest = pathlib.Path(override_path or MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 50 * 1024 * 1024:  # > 50MB sanity
        return str(dest.resolve())

    # Download with headers + retries
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": UA["User-Agent"],
    }

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
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)

            size = dest.stat().st_size
            if size < 50 * 1024 * 1024:
                # Too small -> probably HTML or error page; remove and retry
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise RuntimeError(f"Downloaded model too small: {size} bytes")

            print(f"[nudity] Model saved to {dest} ({size} bytes)")
            break
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed to download ONNX model: {e}") from e
            print(f"[nudity] Download failed (attempt {attempt}), retrying...")

    return str(dest.resolve())

# --------------------------- Helpers ---------------------------

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

    Returns on error:
      {"error": "...", "status": <code>}
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid_path: Optional[str] = None
    frame_paths: List[str] = []

    try:
        # 1) Ensure the ONNX model exists locally (use override path if provided)
        model_file = _ensure_model(model_path)
        print(f"[nudity] Using model: {model_file}")

        # 2) Download video & 3) Extract frames
        vid_path = _download_video(video_url)
        frame_paths = _extract_frames(vid_path)

        # 4) Run detection (batch) with NudeNet ONNX
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
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        _cleanup_files(frame_paths, extra_paths=[vid_path])
