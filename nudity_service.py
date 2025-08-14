# nudity_service.py
import os
import math
import time
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
THRESHOLD         = 0.5
MAX_VIDEO_BYTES   = 50 * 1024 * 1024  # 50 MB

# Target coverage: ~1 frame every 0.5s, up to 60 frames
TARGET_STEP_SEC   = 0.5
MAX_FRAMES        = 60

# Hard time budget to avoid Heroku router timeout (keep < 30s)
TIME_BUDGET_SEC   = 25.0
WARMUP_FRAMES     = 4   # frames to time before auto-thinning

# Resolution: a bit lighter to keep latency predictable
INFER_RES         = 512  # use 640 if you really want, but 512 is safer on Hobby

UA = {"User-Agent": "nudity-service/1.0"}

MODEL_PATH = "/tmp/models/nudenet.onnx"
MODEL_URL  = "https://www.dropbox.com/scl/fi/necew6rgmauez1pnpjyvq/640m.onnx?rlkey=cfu0sr4f6fk3gcrmrdu9u2otp&st=3ny9yqe4&dl=1"

# --------------------------- Model ensure ---------------------------

def _ensure_model(override_path: Optional[str] = None) -> str:
    dest = pathlib.Path(override_path or MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 50 * 1024 * 1024:
        return str(dest.resolve())
    headers = {"Accept": "application/octet-stream", "User-Agent": UA["User-Agent"]}
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 4):
        try:
            print(f"[nudity] Downloading ONNX model to {dest} (attempt {attempt}) ...")
            with requests.get(MODEL_URL, stream=True, timeout=300, headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_iterable = r.iter_content  # silence type hints
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
            if attempt == 3:
                raise RuntimeError(f"Failed to download ONNX model: {e}") from e
            print("[nudity] Download failed, retrying...")
    return str(dest.resolve())

# --------------------------- Singleton detector ---------------------------

_DETECTOR: Optional[NudeDetector] = None
_DETECTOR_LOCK = threading.Lock()
_MODEL_FILE: Optional[str] = None

def _get_detector(override_path: Optional[str] = None) -> NudeDetector:
    global _DETECTOR, _MODEL_FILE
    if _DETECTOR is None:
        with _DETECTOR_LOCK:
            if _DETECTOR is None:
                _MODEL_FILE = _ensure_model(override_path)
                _DETECTOR = NudeDetector(model_path=_MODEL_FILE, inference_resolution=INFER_RES)
                print(f"[nudity] NudeDetector initialized (res={INFER_RES}) with model {_MODEL_FILE}")
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
                raise RuntimeError("Video too large (> 50 MB)")
            f.write(chunk)
        print(f"[nudity] Video saved to {f.name} ({total} bytes)")
        return f.name

def _target_indices(total_frames: int, fps: float) -> List[int]:
    """Evenly spaced indices at ~TARGET_STEP_SEC, capped at MAX_FRAMES, excluding exact endpoints."""
    if total_frames <= 0:
        return []
    fps = fps or 30.0
    duration = total_frames / fps
    desired = max(1, min(MAX_FRAMES, int(math.ceil(duration / TARGET_STEP_SEC))))
    # positions (i+1)/(desired+1): avoids 0 and last frame
    idxs = [int(round(total_frames * (i + 1) / (desired + 1))) for i in range(desired)]
    idxs = sorted({max(0, min(total_frames - 1, i)) for i in idxs})
    return idxs

def _thin_for_budget(remaining_idxs: List[int], processed: int, elapsed: float, avg_secs: float) -> List[int]:
    """Given measured avg per-frame time, thin remaining indices to fit TIME_BUDGET_SEC."""
    budget_left = max(0.0, TIME_BUDGET_SEC - elapsed)
    if avg_secs <= 0:
        return remaining_idxs
    max_more = int(budget_left // avg_secs)
    if max_more <= 0:
        return []
    if len(remaining_idxs) <= max_more:
        return remaining_idxs
    # keep roughly max_more items spread evenly
    step = max(1, len(remaining_idxs) // max_more)
    thinned = remaining_idxs[::step][:max_more]
    return thinned

def _analyze_preds(frame_idx: int, preds: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    hits = []
    for det in preds or []:
        label = det.get("label", det.get("class"))
        score = float(det.get("score", 0.0))
        if label in NUDITY_TAGS and score >= THRESHOLD:
            hits.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / (fps or 30.0), 3),
                "type": label,
                "confidence": score,
            })
    return hits

# --------------------------- Public API ---------------------------

def process_video(video_url: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Download -> pick indices (~0.5s step, ≤60) -> per-frame detect with time-guard -> summarize.
    """
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing video_url", "status": 400}

    vid_path: Optional[str] = None
    hits: List[Dict[str, Any]] = []

    start = time.time()
    try:
        detector = _get_detector(model_path)
        vid_path = _download_video(video_url)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return {"error": "Could not open video", "status": 400}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return {"error": "No frames in video", "status": 400}

        indices = _target_indices(total, fps)
        print(f"[nudity] Planned frames: {len(indices)} (fps={fps:.2f}, total={total})")

        processed = 0
        elapsed = 0.0
        avg_per_frame = None

        i = 0
        while i < len(indices):
            idx = indices[i]

            # hard time guard
            elapsed = time.time() - start
            if elapsed >= TIME_BUDGET_SEC:
                print(f"[nudity] Time budget reached ({elapsed:.2f}s). Stopping early.")
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                i += 1
                continue

            t0 = time.time()
            fd, frame_path = tempfile.mkstemp(suffix=".jpg", dir="/tmp"); os.close(fd)
            cv2.imwrite(frame_path, frame)
            try:
                preds = detector.detect(frame_path)
                hits.extend(_analyze_preds(idx, preds, fps))
            finally:
                try: os.remove(frame_path)
                except Exception: pass
            t1 = time.time()

            processed += 1
            # maintain rolling avg after warmup
            if processed <= WARMUP_FRAMES:
                # simple avg over warmup
                avg_per_frame = ((avg_per_frame or 0.0) * (processed - 1) + (t1 - t0)) / processed
            else:
                # exponential moving average to react to changes
                alpha = 0.2
                avg_per_frame = (1 - alpha) * (avg_per_frame or (t1 - t0)) + alpha * (t1 - t0)

            # after warmup, re-thin remaining frames if needed
            if processed == WARMUP_FRAMES:
                remaining = indices[i+1:]
                thinned = _thin_for_budget(remaining, processed, time.time() - start, avg_per_frame or 0.0)
                if len(thinned) < len(remaining):
                    print(f"[nudity] Auto-thinning remaining frames from {len(remaining)} -> {len(thinned)} to meet time budget.")
                    indices = indices[:i+1] + thinned

            i += 1

        cap.release()

        return {
            "status": 200,
            "nudity_detected": len(hits) > 0,
            "details": hits,                         # frame index, timestamp, label, confidence
            "frames_analyzed": processed,
            "planned_frames": len(indices),
            "elapsed_sec": round(time.time() - start, 3),
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
        det = _get_detector()
        return {"ok": True, "model": _MODEL_FILE or MODEL_PATH, "resolution": INFER_RES}
    except Exception as e:
        return {"ok": False, "error": str(e)}
