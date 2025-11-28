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

# CORS for Vercel (frontend se call ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model load (Vercel serverless mein ek baar load ho jayega)
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLOv12 Rice API - Live on Vercel!"}

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

# Video endpoint (Vercel pe short videos ke liye, lambi mat daal)
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
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
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 50:  # Limit to 50 frames for Vercel timeout
            break
        results = model(frame, conf=0.25, verbose=False)[0]
        out.write(results.plot())
        frame_count += 1

    cap.release()
    out.release()
    os.unlink(tmp_in.name)

    # Streaming response for Vercel (FileResponse se better)
    def iterfile():
        with open(tmp_out, mode="rb") as file_like:
            yield from file_like
        os.unlink(tmp_out)  # Cleanup

    return StreamingResponse(iterfile(), media_type="video/mp4", filename="result.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)