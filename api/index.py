from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os
import tempfile

app = FastAPI(title="YOLOv12 Rice Panicle Detection - Vercel Free")

# Allow frontend se call karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load (root se ../best.pt)
print("Loading YOLOv12 model...")
model = YOLO("../best.pt")  # Vercel free pe CPU pe chalega
print("Model loaded successfully!")

@app.get("/")
def home():
    return {"message": "YOLOv12 Rice Detection API Live on Vercel Free!", "endpoints": ["/predict", "/predict_video"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    results = model(img, conf=0.25, verbose=False)[0]
    annotated = results.plot()
    _, buf = cv2.imencode(".jpg", annotated)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Temp files
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(contents)
    tmp_in.close()

    tmp_out = tmp_in.name.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        os.unlink(tmp_in.name)
        return {"error": "Cannot open video"}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))

    frame_count = 0
    max_frames = 60  # Vercel 60 sec timeout ke andar safe rahega (short videos only)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        results = model(frame, conf=0.25, verbose=False)[0]
        out.write(results.plot())
        frame_count += 1

    cap.release()
    out.release()
    os.unlink(tmp_in.name)

    def iterfile():
        with open(tmp_out, "rb") as f:
            yield from f
        os.unlink(tmp_out)  # cleanup

    return StreamingResponse(iterfile(), media_type="video/mp4", headers={"Content-Disposition": "attachment; filename=result.mp4"})