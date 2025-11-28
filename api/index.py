from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os
import tempfile
import uvicorn

app = FastAPI(title="YOLOv12 Rice Panicle Detection")

# CORS enable (Vercel frontend calls ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load (best.pt root mein hai, Vercel se access ho jayega)
try:
    model = YOLO("../best.pt")  # ../ kyunki api/ folder se root pe jaana hai
except Exception as e:
    print(f"Model load error: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "YOLOv12 Rice API - Live on Vercel! Use /predict for images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded"}
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
    if not model:
        return {"error": "Model not loaded"}
    contents = await file.read()
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(contents)
    tmp_in.close()

    tmp_out = tmp_in.name.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        os.unlink(tmp_in.name)
        return {"error": "Invalid video"}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(tmp_out, fourcc, 20.0, (width, height))

    frame_count = 0
    max_frames = 50  # Vercel timeout ke liye limit (short videos only)
    while True:
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
        os.unlink(tmp_out)  # Cleanup

    return StreamingResponse(iterfile(), media_type="video/mp4", filename="result.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)