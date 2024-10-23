from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from model import predict
from prometheus_client import generate_latest, start_http_server
import numpy as np
import cv2
from PIL import Image
import io
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Enable CORS to allow frontend to communicate with the backend (if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start the Prometheus metrics server
#start_http_server(8001)  # Separate port for Prometheus



# Serve static files (HTML, CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/metrics")
def get_metrics():
    # Expose metrics for Prometheus scraping
    return generate_latest()

@app.post("/detect/")
async def predict_image_detection(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        print("Image loaded!")
        image_with_boxes = predict(image, task='detection')
        img_encoded = cv2.imencode('.jpg', image_with_boxes)[1].tobytes()

        return StreamingResponse(io.BytesIO(img_encoded), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/segment/")
async def predict_image_segmentation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        print("Image loaded!")
        image_with_boxes = predict(image, task='segmentation')
        img_encoded = cv2.imencode('.jpg', image_with_boxes)[1].tobytes()

        return StreamingResponse(io.BytesIO(img_encoded), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
