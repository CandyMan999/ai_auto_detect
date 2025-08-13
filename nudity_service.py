# nudity_service.py
import os
import cv2
import requests
import tempfile
import pathlib

from typing import List, Dict, Any, Optional

# NudeNet imported only when used
from nudenet import NudeDetector

# Match your original tags/threshold
NUDITY_TAGS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}
THRESHOLD = float(os.getenv("NUDITY_THRESHOLD", "0.5"))

MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(50 * 1024 * 1024)))  # 50MB cap
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "9"))  # 10%,20%,...,90% = 9 frames

UA = {"User-Agent": "nudity-service/1.0"}

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
    r = requests.get(url, stream=True, timeout=180, headers=UA, allow_redirects=True)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_VIDEO_BYTES:
                f.close()
                os.unlink(f.name)
                raise ValueError(f"Video too large (max {MAX_VIDEO_BYTES // (1024*1024)} MB)")
            f.write(chunk)
        return f.name

def _extract_frames(video_path: str, total_frames: Optional[int] = None) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    if total_frames is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        total_frames = 100  # fallback

    # 10%, 20%, ..., 90%
    indices = [int(total_frames * i / 10) for i in range(1, 10)]

    tmpdir = tempfile.mkdtemp(prefix="frames_")
    paths: List[str] = []
    try:
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            p = os.path.join(tmpdir, f"frame_{i}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            if len(paths) >= MAX_FRAMES:
                break
    finally:
        cap.release()
    return paths

def _cleanup(files: List[str]):
    for p in files:
        try: os.remove(p)
        except Exception: pass
    # remove parent dirs if empty
    for d in {pathlib.Path(p).parent for p in files}:
        try: os.rmdir(d)
        except Exception: pass

def process_video(video_url: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Pure function used by FastAPI sync route or RQ worker.
    Mirrors your working script: download -> sample frames -> detect_batch -> analyze -> cleanup.
    Returns dict; on error includes {"error": "...", "status": <code>}
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid = None
    frames: List[str] = []
    try:
        vid = _download_video(video_url)
        frames = _extract_frames(vid)
        if not frames:
            return {"error": "No frames extracted", "status": 400}

        # Same as your code: prefer supplied ONNX path, else NudeNet default
        if model_path:
            detector = NudeDetector(model_path=model_path, inference_resolution=640)
        else:
            detector = NudeDetector()  # default model

        batch = detector.detect_batch(frames)
        summary = _analyze(batch)

        return {
            "status": 200,
            "nudity_detected": summary["nudity_detected"],
            "details": summary["details"],
            "frames_analyzed": len(frames),
        }
    except requests.HTTPError as he:
        return {"error": f"Download failed: {he}", "status": 400}
    except ValueError as ve:
        return {"error": str(ve), "status": 413 if "too large" in str(ve).lower() else 400}
    except Exception as e:
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        if vid and os.path.exists(vid):
            try: os.remove(vid)
            except Exception: pass
        if frames:
            _cleanup(frames)
