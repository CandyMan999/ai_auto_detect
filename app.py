import io
import os
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# --- HEIC/HEIF support for iPhone ---
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    # If the wheel isn't available, uploads still work for JPEG/PNG/WEBP.
    pass

# --- Config ---
MAX_BYTES = 10 * 1024 * 1024   # bump cap to 10 MB for iPhone photos
REQUIRE_API_KEY = os.getenv("API_KEY")  # set on Heroku to require x-api-key

app = FastAPI(title="Face Gate (OpenCV + HEIC)")

# CORS: open by default; tighten to your domains later if desired
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["https://yourapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Haar cascade (no libGL)
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
async def detect_face(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    # Decode image (don’t trust content-type; HEIC handled by pillow-heif)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="Upload a picture with your face")

    return {"faces": int(len(faces)), "is_face": True}
