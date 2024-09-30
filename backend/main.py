from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from model import predict
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Serve static files (HTML, CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS to allow frontend to communicate with the backend (if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        
        # Convert the contents to a NumPy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode the image from the NumPy array
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Verify that the image was loaded properly
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        print("Image loaded!")

        # Perform prediction and get the image with overlaid boxes
        image_with_boxes = predict(image)
        im = Image.fromarray(image_with_boxes)
        im.save("your_file.jpeg")
        print("Predicted!")

        # Convert the image with boxes back to a byte stream
        _, img_encoded = cv2.imencode('.jpg', image_with_boxes)
        img_byte_arr = io.BytesIO(img_encoded.tobytes())
        img_byte_arr.seek(0)

        # Return the image with bounding boxes as a StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
