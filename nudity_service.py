# nudity_service.py
import os
import cv2
import requests
import tempfile
import pathlib
from typing import List, Dict, Any

from nudenet import NudeDetector  # ONNX inference

NUDITY_TAGS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}
THRESHOLD = 0.5
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB cap
MAX_FRAMES = 9
UA = {"User-Agent": "nudity-service/1.0"}

# Direct model URL
MODEL_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx"
MODEL_PATH = "/tmp/nudenet.onnx"


def _ensure_model() -> str:
    """Always download ONNX model to /tmp if missing."""
    p = pathlib.Path(MODEL_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and p.stat().st_size > 1 * 1024 * 1024:
        return str(p.resolve())

    print(f"Downloading NudeNet model from {MODEL_URL}...")
    with requests.get(MODEL_URL, stream=True, timeout=300, headers=UA) as r:
        r.raise_for_status()
        with open(p, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    if p.stat().st_size < 5 * 1024 * 1024:
        raise RuntimeError("Downloaded model file is too small or corrupted")

    return str(p.resolve())


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
    r = requests.get(url, stream=True, timeout=300, headers=UA)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_VIDEO_BYTES:
                f.close()
                os.unlink(f.name)
                raise RuntimeError(f"Video too large (> {MAX_VIDEO_BYTES // (1024*1024)} MB)")
            f.write(chunk)
        return f.name


def _extract_frames(video_path: str) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError("No frames in video")

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
    for d in {pathlib.Path(p).parent for p in files}:
        try:
            os.rmdir(d)
        except Exception:
            pass


def process_video(video_url: str) -> Dict[str, Any]:
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid = None
    frames: List[str] = []
    try:
        model_file = _ensure_model()
        detector = NudeDetector(model_path=model_file, inference_resolution=640)

        vid = _download_video(video_url)
        frames = _extract_frames(vid)

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
        msg = str(re)
        code = 413 if "too large" in msg.lower() else 400
        return {"error": msg, "status": code}
    except Exception as e:
        return {"error": f"Detection failed: {e}", "status": 500}
    finally:
        if vid and os.path.exists(vid):
            try:
                os.remove(vid)
            except Exception:
                pass
        if frames:
            _cleanup_files(frames)
