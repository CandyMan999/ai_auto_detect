# nudity_service.py
import os
import cv2
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

THRESHOLD            = float(os.getenv("NUDITY_THRESHOLD", "0.5"))
MAX_VIDEO_BYTES      = int(os.getenv("MAX_VIDEO_BYTES", str(50 * 1024 * 1024)))  # 50MB

# Sampling knobs (all overridable via request body too)
SAMPLING_MODE        = os.getenv("SAMPLING_MODE", "uniform")  # uniform | stride | fps | deciles
FRAME_STRIDE         = int(os.getenv("FRAME_STRIDE", "5"))    # used when SAMPLING_MODE=stride
TARGET_FRAMES        = int(os.getenv("TARGET_FRAMES", "60"))  # used when SAMPLING_MODE=uniform
SECONDS_STRIDE       = float(os.getenv("SECONDS_STRIDE", "0.5"))  # used when SAMPLING_MODE=fps (seconds between samples)
MAX_SAMPLED_FRAMES   = int(os.getenv("MAX_SAMPLED_FRAMES", "120"))  # hard cap for safety

INFER_RES            = int(os.getenv("INFER_RES", "512"))     # NudeNet inference resolution

UA = {"User-Agent": "nudity-service/1.0"}

# Writable path on Heroku; re-downloads on dyno restart
MODEL_PATH = os.getenv("NUDENET_MODEL_PATH", "/tmp/models/nudenet.onnx")
MODEL_URL  = os.getenv("NUDENET_MODEL_URL")  # Dropbox/S3 direct link

# --------------------------- Model ensure ---------------------------

def _ensure_model(override_path: Optional[str] = None) -> str:
    dest = pathlib.Path(override_path or MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 50 * 1024 * 1024:
        return str(dest.resolve())

    if not MODEL_URL:
        raise RuntimeError("NUDENET_MODEL_URL not set and model file missing")

    headers = {"Accept": "application/octet-stream", "User-Agent": UA["User-Agent"]}
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, 5):
        try:
            print(f"[nudity] Downloading ONNX model to {dest} (attempt {attempt}) ...")
            with requests.get(MODEL_URL, stream=True, timeout=300, headers=headers, allow_redirects=True) as r:
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
                _DETECTOR = NudeDetector(model_path=_MODEL_FILE, inference_resolution=INFER_RES)
                print(f"[nudity] NudeDetector initialized once with model {_MODEL_FILE} (res={INFER_RES})")
    return _DETECTOR

# --------------------------- Helpers ---------------------------

def _download_video(url: str) -> str:
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

def _unique_sorted(ints: List[int]) -> List[int]:
    return sorted(set(i for i in ints if i >= 0))

def _indices_deciles(total: int) -> List[int]:
    return [int(total * i / 10) for i in range(1, 10)]

def _indices_uniform(total: int, target: int) -> List[int]:
    target = max(1, min(target, total))
    # spread across (not including 0 or last) similar to deciles but denser
    return [int((i+1) * total / (target + 1)) for i in range(target)]

def _indices_stride(total: int, stride: int) -> List[int]:
    stride = max(1, stride)
    return list(range(0, total, stride))

def _indices_fps(total: int, fps: float, seconds_stride: float) -> List[int]:
    step = max(1, int(round(fps * max(0.01, seconds_stride))))
    return list(range(0, total, step))

def _choose_indices(total_frames: int, fps: float,
                    sampling_mode: str, frame_stride: int,
                    target_frames: int, seconds_stride: float) -> List[int]:
    mode = (sampling_mode or SAMPLING_MODE).lower()
    if mode == "deciles":
        idx = _indices_deciles(total_frames)
    elif mode == "stride":
        idx = _indices_stride(total_frames, frame_stride or FRAME_STRIDE)
    elif mode == "fps":
        idx = _indices_fps(total_frames, fps, seconds_stride or SECONDS_STRIDE)
    else:  # "uniform" (default)
        idx = _indices_uniform(total_frames, target_frames or TARGET_FRAMES)

    idx = _unique_sorted(idx)
    if MAX_SAMPLED_FRAMES > 0 and len(idx) > MAX_SAMPLED_FRAMES:
        # uniformly thin to the cap
        step = max(1, len(idx) // MAX_SAMPLED_FRAMES)
        idx = idx[::step][:MAX_SAMPLED_FRAMES]
    return idx

def _analyze_frame_preds(frame_idx: int, preds: List[Dict[str, Any]], fps: float, threshold: float) -> List[Dict[str, Any]]:
    hits = []
    for det in preds or []:
        label = det.get("label", det.get("class"))
        score = float(det.get("score", 0.0))
        if label in NUDITY_TAGS and score >= threshold:
            hits.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / (fps or 30.0), 3),
                "type": label,
                "confidence": score,
            })
    return hits

# --------------------------- Public API ---------------------------

def process_video(
    video_url: str,
    model_path: Optional[str] = None,
    sampling_mode: Optional[str] = None,
    frame_stride: Optional[int] = None,
    target_frames: Optional[int] = None,
    seconds_stride: Optional[float] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:

    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid_path: Optional[str] = None
    hits: List[Dict[str, Any]] = []
    thr = float(threshold if threshold is not None else THRESHOLD)

    try:
        detector = get_detector(model_path)

        vid_path = _download_video(video_url)
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return {"error": "Could not open video", "status": 400}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return {"error": "No frames in video", "status": 400}

        indices = _choose_indices(
            total_frames=total,
            fps=fps,
            sampling_mode=sampling_mode or SAMPLING_MODE,
            frame_stride=frame_stride or FRAME_STRIDE,
            target_frames=target_frames or TARGET_FRAMES,
            seconds_stride=seconds_stride or SECONDS_STRIDE,
        )

        print(f"[nudity] Sampling mode={sampling_mode or SAMPLING_MODE}, "
              f"indices={len(indices)}, fps={fps:.2f}, total_frames={total}")

        # memory-safe loop per frame
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fd, frame_path = tempfile.mkstemp(suffix=".jpg", dir="/tmp")
            os.close(fd)
            cv2.imwrite(frame_path, frame)
            try:
                preds = detector.detect(frame_path)
                hits.extend(_analyze_frame_preds(idx, preds, fps, thr))
            finally:
                try: os.remove(frame_path)
                except Exception: pass

        cap.release()

        return {
            "status": 200,
            "nudity_detected": len(hits) > 0,
            "details": hits[:500],          # cap payload
            "frames_analyzed": len(indices)
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
        try:
            if vid_path and os.path.exists(vid_path):
                os.remove(vid_path)
        except Exception:
            pass

def nudity_health() -> Dict[str, Any]:
    try:
        det = get_detector()
        return {
            "ok": True,
            "model": _MODEL_FILE or MODEL_PATH,
            "resolution": getattr(det, "inference_resolution", "unknown")
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
