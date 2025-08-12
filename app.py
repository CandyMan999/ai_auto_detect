import io
import os
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header

# --- Config ---
MAX_BYTES = 5 * 1024 * 1024
REQUIRE_API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Face Gate (OpenCV)")

# Haar cascade from OpenCV package
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _check_api_key(x_api_key: str | None):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/detect")
async def detect_face(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB)")

    # Decode with Pillow (don’t trust content-type)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    # Convert to grayscale for Haar
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Detect faces (tune params for your tolerance)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,    # 1.05–1.3
        minNeighbors=5,     # 3–8
        minSize=(60, 60)    # ignore tiny detections
    )

    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": int(len(faces)), "is_face": True}
