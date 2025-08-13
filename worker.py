# worker.py — RQ worker job: face detection
import io, os, pathlib, time, requests
import numpy as np
from PIL import Image
import cv2

# OPTIONAL HEIC
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

# ---------- Model bootstrap (same as app.py) ----------
MODEL_DIR = pathlib.Path("./face_model"); MODEL_DIR.mkdir(exist_ok=True)
PROTO_PATH = MODEL_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
URL_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
URL_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def _download(url, dest, min_bytes):
    for i in range(1,5):
        try:
            with requests.get(url, timeout=90, stream=True, headers={"User-Agent":"face-gate/1.0"}) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1<<20):
                        if chunk: f.write(chunk)
            if dest.stat().st_size < min_bytes:
                raise RuntimeError("too small")
            return
        except Exception:
            if dest.exists():
                try: dest.unlink()
                except: pass
            if i == 4: raise
            time.sleep(1.5*i)

if not PROTO_PATH.exists():   _download(URL_PROTO, PROTO_PATH, 1*1024)
if not WEIGHTS_PATH.exists(): _download(URL_WEIGHTS, WEIGHTS_PATH, 5*1024*1024)

NET = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(WEIGHTS_PATH))

def detect_faces_bytes(
    data: bytes,
    conf_thresh: float = 0.65,
    max_bytes: int = 10 * 1024 * 1024
) -> dict:
    if not data:
        return {"error": "No file uploaded", "status": 400}
    if len(data) > max_bytes:
        return {"error": "Image too large", "status": 413}
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"error": "File is not a valid image", "status": 400}
    arr = np.array(img)
    blob = cv2.dnn.blobFromImage(arr, 1.0, (300,300), (104.0,177.0,123.0), swapRB=True, crop=False)
    NET.setInput(blob)
    detections = NET.forward()
    count = sum(1 for i in range(detections.shape[2]) if float(detections[0,0,i,2]) >= conf_thresh)
    if count == 0:
        return {"error": "Upload a picture with your face", "status": 422}
    return {"faces": int(count), "is_face": True, "status": 200}
