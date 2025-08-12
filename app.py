import io
import os
import numpy as np
from PIL import Image
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException, Header

# --- Optional HEIC/HEIF support (safe if not installed) ---
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    pass

# --- Config ---
MAX_BYTES = 5 * 1024 * 1024  # 5 MB cap
REQUIRE_API_KEY = os.getenv("API_KEY")  # set on Heroku to enable key check

# --- App & detector ---
app = FastAPI(title="Face Gate")
mp_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,               # short-range; ideal for profile pics
    min_detection_confidence=0.6
)

def _check_api_key(x_api_key: str | None):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    _check_api_key(x_api_key)

    # Read bytes (don’t trust content_type header)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB)")

    # Try to decode as image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        # True binary sniffing via Pillow failed => not an image
        raise HTTPException(status_code=400, detail="File is not a valid image")

    # Face detection
    arr = np.array(img)  # RGB
    res = mp_detector.process(arr)
    count = 0 if res.detections is None else len(res.detections)

    if count == 0:
        # exact message requested
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": count, "is_face": True}
