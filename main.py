import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# mediapipe.framework.formats import landmark_pb2 removed for python3.12 compatibility

app = FastAPI(title="Human Pose Estimation API", version="1.1.0")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

# Download model if it doesn't exist locally (useful for local testing)
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# Initialize the PoseLandmarker using the Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# Drawing utilities to render the landmarks
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.post("/predict")
async def predict_pose(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image file."}

    # Convert the BGR image to RGB (MediaPipe requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Run the model
    detection_result = detector.detect(mp_image)
    
    # Draw the pose annotations on the image manually to avoid protobuf dependency
    if detection_result.pose_landmarks:
        height, width, _ = image.shape
        for pose_landmarks in detection_result.pose_landmarks:
            # Draw connections first so they appear behind landmarks
            if mp_pose.POSE_CONNECTIONS:
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_lm = pose_landmarks[start_idx]
                    end_lm = pose_landmarks[end_idx]
                    
                    start_point = (int(start_lm.x * width), int(start_lm.y * height))
                    end_point = (int(end_lm.x * width), int(end_lm.y * height))
                    
                    cv2.line(image, start_point, end_point, (245, 117, 66), 2)
            
            # Draw landmarks
            for landmark in pose_landmarks:
                point = (int(landmark.x * width), int(landmark.y * height))
                cv2.circle(image, point, 2, (245, 66, 230), -1)
            
    # Encode image back to bytes
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return {"error": "Failed to encode the processed image."}
        
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Human Pose Estimation API (Tasks API Vision). Use the /predict endpoint to process images."}
