import io, os
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# --- HEIC/HEIF support for iPhone ---
try:
    from pillow_heif import register_heif_opener  # optional
    register_heif_opener()
except Exception:
    pass

MAX_BYTES = 10 * 1024 * 1024
REQUIRE_API_KEY = os.getenv("API_KEY")
CONF_THRESH = float(os.getenv("FACE_CONF", "0.6"))   # <— tune sensitivity here (0.5–0.8 good range)

app = FastAPI(title="Face Gate (OpenCV DNN)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_api_key(x_api_key: str | None):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Load DNN model once (Caffe)
MODEL_PROTO = os.getenv("FACE_PROTO", "deploy.prototxt")
MODEL_WEIGHTS = os.getenv("FACE_MODEL", "res10_300x300_ssd_iter_140000.caffemodel")
if not (os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS)):
    raise RuntimeError("Face model files not found. Ensure deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel are present.")
net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

@app.get("/health")
def health():
    return {"ok": True, "conf": CONF_THRESH}

@app.post("/detect")
async def detect_face(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    # Decode (auto HEIC/JPEG/PNG/WEBP)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    arr = np.array(img)
    (h, w) = arr.shape[:2]

    # DNN blob: 300x300, mean subtraction (as per model)
    blob = cv2.dnn.blobFromImage(arr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    count = 0
    # detections shape: [1,1,N,7] => [batch, class, idx, (id, conf, x1,y1,x2,y2)]
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= CONF_THRESH:
            count += 1

    if count == 0:
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": count, "is_face": True}
