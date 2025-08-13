# app.py
import io
import os
import pathlib
import time
import requests
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# --- Optional HEIC/HEIF support (iPhone) ---
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    pass

# ---------------- Config ----------------
MAX_BYTES = int(os.getenv("MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
REQUIRE_API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
CONF_THRESH = float(os.getenv("FACE_CONF", "0.65"))  # 0.5 permissive … 0.8 strict

# ---------------- App ----------------
app = FastAPI(title="Face Gate (OpenCV DNN)")

# CORS: open by default; narrow to your domains later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g., ["https://yourapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_api_key(x_api_key: str | None):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------- Model download & load (robust) ----------------
MODEL_DIR = pathlib.Path("./face_model")
MODEL_DIR.mkdir(exist_ok=True)

PROTO_PATH = MODEL_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

URL_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
URL_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

UA = {"User-Agent": "face-gate/1.0"}  # helps avoid some CDN issues

def _download_with_retries(url: str, dest: pathlib.Path, attempts: int = 4, min_bytes: int = 1024):
    for i in range(1, attempts + 1):
        try:
            with requests.get(url, timeout=90, stream=True, headers=UA, allow_redirects=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            # sanity check
            if dest.stat().st_size < min_bytes:
                raise RuntimeError(f"Downloaded file too small: {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception:
            if dest.exists():
                try: dest.unlink()
                except Exception: pass
            if i == attempts:
                raise
            time.sleep(1.5 * i)  # backoff

def ensure_face_model():
    # prototxt is small (~30–40 KB) ⇒ require >1 KB
    if not PROTO_PATH.exists():
        _download_with_retries(URL_PROTO, PROTO_PATH, min_bytes=1 * 1024)
    # caffemodel is ~10–11 MB ⇒ require >5 MB
    if not WEIGHTS_PATH.exists():
        _download_with_retries(URL_WEIGHTS, WEIGHTS_PATH, min_bytes=5 * 1024 * 1024)

ensure_face_model()

# Load DNN model once (Caffe)
net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(WEIGHTS_PATH))

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "conf": CONF_THRESH}

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_BYTES // (1024*1024)} MB)")

    # Decode (don’t trust content-type; HEIC supported if pillow-heif is present)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    arr = np.array(img)  # HxWx3 RGB
    blob = cv2.dnn.blobFromImage(arr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()  # shape [1,1,N,7]  => [id, conf, x1,y1,x2,y2]

    count = 0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= CONF_THRESH:
            count += 1

    if count == 0:
        # Exact message required
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": count, "is_face": True}
