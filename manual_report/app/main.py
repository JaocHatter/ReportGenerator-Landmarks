from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from typing import List
from app.schemas import Landmark, LandmarkMetadata, PoseData
from app.gemini_service import GeminiService
from app.report_generator import ReportGenerator
import time
from contextlib import asynccontextmanager
import json

OUTPUT_DIR = "output"
LANDMARK_IMAGE_DIR = os.path.join(OUTPUT_DIR, "landmark_images")
TRAJECTORY_DATA_DIR = os.path.join(OUTPUT_DIR, "trajectory_data")
TRAJECTORY_FILE = os.path.join(TRAJECTORY_DATA_DIR, "path.txt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup logic."""
    os.makedirs(LANDMARK_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAJECTORY_DATA_DIR, exist_ok=True)
    
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    app.state.gemini_service = GeminiService(api_key=GEMINI_API_KEY)
    app.state.landmarks = [] 
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

@app.post("/add_landmark/", response_model=Landmark, status_code=201)
async def add_landmark(
    request: Request,
    image: UploadFile = File(...),
    metadata_json: str = Form(...)
):
    """Receives landmark data, analyzes it, and stores it."""
    try:
        metadata = LandmarkMetadata(**json.loads(metadata_json))
        image_bytes = await image.read()
        
        gemini_service: GeminiService = request.app.state.gemini_service
        analysis_result = await gemini_service.get_contextual_analysis(image_bytes)

        if not analysis_result or not analysis_result.get("object_name"):
            raise HTTPException(status_code=400, detail="Could not identify a valid landmark name.")

        landmark_id = f"LM_{int(time.time())}"
        image_filename = f"{landmark_id}{os.path.splitext(image.filename)[1] or '.jpg'}"
        image_filepath = os.path.join(LANDMARK_IMAGE_DIR, image_filename)

        with open(image_filepath, "wb") as buffer:
            buffer.write(image_bytes)

        new_landmark = Landmark(
            id=landmark_id,
            name=analysis_result["object_name"],
            location=metadata.position,
            timestamp=metadata.timestamp,
            best_image_path=os.path.abspath(image_filepath),
            detailed_description=analysis_result["description"],
            contextual_analysis=analysis_result["analysis"]
        )

        request.app.state.landmarks.append(new_landmark.model_dump()) # Store as dict
        logger.info(f"New landmark added: {new_landmark.name} (ID: {new_landmark.id})")
        return new_landmark
    except Exception as e:
        logger.error(f"Unexpected error in add_landmark: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/add_pose/", status_code=201)
async def add_pose(data: PoseData):
    """Receives rover pose data and appends it to the trajectory file."""
    try:
        pos = data.pose.position
        with open(TRAJECTORY_FILE, "a") as f:
            f.write(f"{pos.x},{pos.y}\n")
        logger.info(f"Pose added to trajectory: X={pos.x}, Y={pos.y}")
        return JSONResponse(content={"status": "success", "message": "Pose data added."})
    except Exception as e:
        logger.error(f"Failed to write pose data: {e}")
        raise HTTPException(status_code=500, detail="Failed to write pose data.")

@app.get("/landmarks/", response_model=List[Landmark])
async def get_all_landmarks(request: Request):
    """Returns all stored landmark data."""
    return request.app.state.landmarks

@app.get("/generate_report/", response_class=FileResponse)
async def generate_full_report(request: Request):
    """
    Generates a full PDF mission report and returns it as a downloadable file.
    """
    landmarks = request.app.state.landmarks
    if not landmarks:
        raise HTTPException(status_code=404, detail="No landmarks available to generate a report.")
    
    try:
        report_generator = ReportGenerator(
            landmarks_data=landmarks,
            trajectory_data_path=TRAJECTORY_FILE
        )
        
        pdf_filepath = report_generator.generate_report()
        
        if not os.path.exists(pdf_filepath):
            raise HTTPException(status_code=500, detail="Report file was not created.")

        return FileResponse(
            path=pdf_filepath,
            filename=os.path.basename(pdf_filepath),
            media_type='application/pdf',
            background=BackgroundTask(os.remove, pdf_filepath) 
        )
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during report generation: {e}")

from starlette.background import BackgroundTask
