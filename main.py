from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os
import tempfile

app = FastAPI(title="YOLOv12 Rice Panicle Detection")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model load — best.pt root mein hoga
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLOv12 Rice API Live on Vercel!", "test_at": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img, conf=0.25, verbose=False)[0]
    annotated = results.plot()
    _, buf = cv2.imencode(".jpg", annotated)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(contents)
    tmp_in.close()

    tmp_out = tmp_in.name.replace(".mp4", "_out.mp4")
    cap = cv2.VideoCapture(tmp_in.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened() and frame_count < 70:  # short video only
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=0.25, verbose=False)[0]
        out.write(results.plot())
        frame_count += 1

    cap.release()
    out.release()
    os.unlink(tmp_in.name)

    return FileResponse(tmp_out, media_type="video/mp4", filename="result.mp4",
                        background=lambda: os.unlink(tmp_out) if os.path.exists(tmp_out) else None)
