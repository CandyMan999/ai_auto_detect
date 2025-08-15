# app.py
import io
import os
import time
import pathlib
import requests
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Queue (optional; used by /detect_async only)
import redis
from rq import Queue

# ------------------------------------------------------------
# Config (ENV)
# ------------------------------------------------------------
MAX_BYTES       = int(os.getenv("MAX_BYTES", str(10 * 1024 * 1024)))   # face image max bytes
CONF_THRESH     = float(os.getenv("FACE_CONF", "0.55"))                # face detector confidence
REQUIRE_API_KEY = os.getenv("API_KEY")                                 # optional; if set, require x-api-key
REDIS_URL       = os.getenv("REDIS_TLS_URL") or os.getenv("REDIS_URL") # optional; for /detect_async

# ------------------------------------------------------------
# FastAPI + CORS
# ------------------------------------------------------------
app = FastAPI(title="Face & Nudity Detection Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_api_key(x_api_key: str | None):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ------------------------------------------------------------
# iPhone HEIC/HEIF support
# ------------------------------------------------------------
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    # If pillow-heif isn't available, we just skip HEIC support
    pass

# ------------------------------------------------------------
# Face model (OpenCV DNN) bootstrap
# ------------------------------------------------------------
MODEL_DIR = pathlib.Path("./face_model")
MODEL_DIR.mkdir(exist_ok=True)

PROTO_PATH   = MODEL_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

URL_PROTO   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
URL_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
UA = {"User-Agent": "face-gate/1.0"}

FRONTAL_CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
PROFILE_CASCADE_PATH = MODEL_DIR / "haarcascade_profileface.xml"
URL_HAAR_FRONTAL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
URL_HAAR_PROFILE = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml"

def _download_with_retries(url: str, dest: pathlib.Path, attempts: int, min_bytes: int):
    for i in range(1, attempts + 1):
        try:
            with requests.get(url, timeout=90, stream=True, headers=UA, allow_redirects=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            if dest.stat().st_size < min_bytes:
                raise RuntimeError(f"Downloaded file too small: {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception:
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            if i == attempts:
                raise
            time.sleep(1.5 * i)

def ensure_face_model():
    if not PROTO_PATH.exists():
        _download_with_retries(URL_PROTO, PROTO_PATH, attempts=4, min_bytes=1 * 1024)
    if not WEIGHTS_PATH.exists():
        _download_with_retries(URL_WEIGHTS, WEIGHTS_PATH, attempts=4, min_bytes=5 * 1024 * 1024)
          # NEW: cascades (small files; ~20–100 KB)
    if not FRONTAL_CASCADE_PATH.exists():
        _download_with_retries(URL_HAAR_FRONTAL, FRONTAL_CASCADE_PATH, attempts=4, min_bytes=2 * 1024)
    if not PROFILE_CASCADE_PATH.exists():
        _download_with_retries(URL_HAAR_PROFILE,  PROFILE_CASCADE_PATH, attempts=4, min_bytes=2 * 1024)

# Download once and load the OpenCV face net
ensure_face_model()
NET = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(WEIGHTS_PATH))
# NEW:
FRONTAL_CASCADE = cv2.CascadeClassifier(str(FRONTAL_CASCADE_PATH))
PROFILE_CASCADE = cv2.CascadeClassifier(str(PROFILE_CASCADE_PATH))

# ------------------------------------------------------------
# Face detection core
# ------------------------------------------------------------
def detect_faces_bytes(data: bytes, conf_thresh: float = CONF_THRESH, max_bytes: int = MAX_BYTES) -> dict:
    if not data:
        return {"error": "No file uploaded", "status": 400}
    if len(data) > max_bytes:
        return {"error": f"Image too large (max {max_bytes // (1024*1024)} MB)", "status": 413}

    # Load image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"error": "File is not a valid image", "status": 400}
    arr = np.array(img)

    # ---------- Primary: DNN (best precision for frontal) ----------
    blob = cv2.dnn.blobFromImage(arr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    NET.setInput(blob)
    detections = NET.forward()  # [1,1,N,7]

    dnn_count = 0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            dnn_count += 1

    if dnn_count > 0:
        return {"faces": int(dnn_count), "is_face": True, "status": 200}

    # ---------- Fallback: Haar cascades (catches profiles) ----------
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Frontal
    frontal = FRONTAL_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )

    # Profile (both left and right via flip)
    profile_left = PROFILE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )
    if len(profile_left) == 0:
        gray_flipped = cv2.flip(gray, 1)
        profile_right = PROFILE_CASCADE.detectMultiScale(
            gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
        profile_count = len(profile_right)
    else:
        profile_count = len(profile_left)

    haar_total = len(frontal) + profile_count
    if haar_total > 0:
        # Optional: include how we succeeded; your client can ignore extra keys
        return {"faces": int(haar_total), "is_face": True, "status": 200, "method": "haar_fallback"}

    # Nothing found
    return {"error": "Upload a picture with your face", "status": 422}


# ------------------------------------------------------------
# Queue helpers (face async)
# ------------------------------------------------------------
def get_queue():
    if not REDIS_URL:
        return None
    try:
        conn = redis.from_url(REDIS_URL, ssl_cert_reqs=None)  # Heroku Redis over TLS
        return Queue("facequeue", connection=conn, default_timeout=30)
    except Exception:
        return None

# ------------------------------------------------------------
# Routes: health + face sync/async
# ------------------------------------------------------------
@app.get("/health")
def health():
    # Import here to avoid any circular import concerns
    from nudity_service import nudity_health
    return {
        "ok": True,
        "conf": CONF_THRESH,
        "queue": bool(get_queue()),
        "nudity": nudity_health()
    }

@app.post("/detect")
async def detect_face_sync(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    data = await file.read()
    result = detect_faces_bytes(data, conf_thresh=CONF_THRESH, max_bytes=MAX_BYTES)
    if result.get("error"):
        raise HTTPException(status_code=result["status"], detail=result["error"])
    return {"faces": result["faces"], "is_face": result["is_face"]}

@app.post("/detect_async")
async def detect_face_async(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    q = get_queue()
    data = await file.read()

    if not q:
        # Fallback to sync if no queue configured
        result = detect_faces_bytes(data, conf_thresh=CONF_THRESH, max_bytes=MAX_BYTES)
        if result.get("error"):
            raise HTTPException(status_code=result["status"], detail=result["error"])
        return {"faces": result["faces"], "is_face": result["is_face"], "mode": "sync_fallback"}

    # Enqueue background job
    job = q.enqueue("app.detect_faces_bytes", data, kwargs={"conf_thresh": CONF_THRESH, "max_bytes": MAX_BYTES}, result_ttl=300, ttl=60)
    return {"job_id": job.get_id(), "status": "queued"}

@app.get("/result/{job_id}")
def get_result(job_id: str, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    q = get_queue()
    if not q:
        raise HTTPException(503, "Queue unavailable")
    job = q.fetch_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.is_finished and not job.is_failed:
        return {"status": "processing"}
    if job.is_failed:
        raise HTTPException(500, "Detection failed")
    result = job.result
    if result.get("error"):
        raise HTTPException(status_code=result["status"], detail=result["error"])
    return {"status": "done", "faces": result["faces"], "is_face": result["is_face"]}

# ------------------------------------------------------------
# Nudity detection route (delegates to nudity_service.py)
# ------------------------------------------------------------
class NudityRequest(BaseModel):
    video_url: str

from nudity_service import process_video  # one-way import

@app.post("/nudity/detect")
async def nudity_detect_sync(req: NudityRequest, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    # optional override; otherwise nudity_service uses /tmp/models/nudenet.onnx
    model_path = os.getenv("NUDENET_MODEL_PATH")
    res = process_video(req.video_url, model_path=model_path)
    if res.get("error"):
        raise HTTPException(status_code=res["status"], detail=res["error"])
    return res

# ------------------------------------------------------------
# Optional root
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "face+nudity"}
