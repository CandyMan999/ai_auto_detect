# app.py
import io
import os
import hashlib
import pathlib
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
    # If pillow-heif isn't installed, JPEG/PNG/WEBP still work.
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

# ---------------- Model download & load ----------------
MODEL_DIR = pathlib.Path("./face_model")
MODEL_DIR.mkdir(exist_ok=True)

PROTO_PATH = MODEL_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

URL_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
URL_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
# Known hash published for this model
WEIGHTS_SHA1 = "15aa726b4d46d9f023526d85537db81cbc8dd566"

def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: pathlib.Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)

def ensure_face_model():
    # prototxt
    if not PROTO_PATH.exists():
        _download(URL_PROTO, PROTO_PATH)
    # weights
    if not WEIGHTS_PATH.exists():
        _download(URL_WEIGHTS, WEIGHTS_PATH)
    # verify sha1 of weights
    if _sha1(WEIGHTS_PATH) != WEIGHTS_SHA1:
        WEIGHTS_PATH.unlink(missing_ok=True)
        raise RuntimeError("Face weights checksum mismatch")

ensure_face_model()

# Load DNN model once
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
    (h, w) = arr.shape[:2]

    # Create blob for DNN: 300x300, mean subtraction, swapRB=True per model
    blob = cv2.dnn.blobFromImage(arr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()  # shape [1,1,N,7]

    count = 0
    # detections[0,0,i] = [id, conf, x1, y1, x2, y2] (x/y are normalized)
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= CONF_THRESH:
            count += 1

    if count == 0:
        # Exact message required
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": count, "is_face": True}
