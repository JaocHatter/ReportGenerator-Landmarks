from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
import os
from typing import List, Dict
import logging
from app.schemas import Landmark, Position, LandmarkMetadata
from app.gemini_service import GeminiService
import time
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import shutil
import json
import uuid
import time

LANDMARK_IMAGE_DIR = "output/landmark_images"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        os.makedirs(LANDMARK_IMAGE_DIR, exist_ok=True)
        GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        
        app.state.gemini_service = GeminiService(api_key=GEMINI_API_KEY)
        logger.info("Gemini service initialized successfully.")

        # temporal place to store landmarks
        app.state.landmarks = []

        logger.info("Gemini service initialized successfully")
        yield
    except ValueError as e:
        logger.error(f"ValueError during application startup: {e}")
    except Exception as e:
        logger.critical(f"Critical error during application startup: {e}")

app = FastAPI(lifespan=lifespan)

@app.post("/add_landmark/", response_model=Landmark, status_code=201)
async def add_landmark(
    image: UploadFile = File(..., description="Image file of the potential landmark."),
    metadata_json: str = Form(..., description="JSON string with landmark metadata (position, timestamp).")
):
    """
    Receives an image, saves it, performs a contextual analysis using Gemini,
    and stores the confirmed landmark's information.
    """
    try:
        metadata = LandmarkMetadata(**json.loads(metadata_json))
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for metadata: {e}")

    try:
        image_bytes = await image.read()
        gemini_service: GeminiService = app.state.gemini_service
        analysis_result = await gemini_service.get_contextual_analysis(image_bytes)

        if not analysis_result or not analysis_result.get("object_name"):
            raise HTTPException(status_code=400, detail="Could not identify a valid landmark name from the image.")

        # --- Save the image file ---
        landmark_id = f"LM_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        file_extension = os.path.splitext(image.filename)[1] or ".jpg"
        image_filename = f"{landmark_id}{file_extension}"
        image_filepath = os.path.join(LANDMARK_IMAGE_DIR, image_filename)

        with open(image_filepath, "wb") as buffer:
            buffer.write(image_bytes)
        
        absolute_image_path = os.path.abspath(image_filepath)
        logger.info(f"Image for landmark {landmark_id} saved to {absolute_image_path}")

        new_landmark = Landmark(
            id=landmark_id,
            name=analysis_result["object_name"],
            location=metadata.position,
            timestamp=metadata.timestamp,
            best_image_path=absolute_image_path, 
            detailed_description=analysis_result["description"],
            contextual_analysis=analysis_result["analysis"]
        )

        app.state.landmarks.append(new_landmark)
        logger.info(f"Added new landmark: {new_landmark.name} (ID: {new_landmark.id})")

        return new_landmark

    except Exception as e:
        logger.error(f"An unexpected error occurred in add_landmark: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/landmarks/", response_model=List[Landmark])
async def get_all_landmarks():
    return app.state.landmarks

@app.delete("/erase_last_landmark/", response_model=Landmark)
async def erase_last_landmark():
    """
    Removes the most recently added landmark from the list.
    """
    if not app.state.landmarks:
        raise HTTPException(status_code=404, detail="No landmarks to erase.")
    
    last_landmark = app.state.landmarks.pop()
    logger.info(f"Erased last landmark: {last_landmark.name} (ID: {last_landmark.id})")
    return last_landmark

@app.get("/generate_report/", response_class=PlainTextResponse)
async def generate_report():
    """
    Generates a simple text-based report of all landmarks, suitable for a quick preview.
    The dedicated script in /scripts/generate_report.py should be used for file-based reports.
    """
    if not app.state.landmarks:
        return "No landmarks have been identified."

    report_lines = [f"# ERC Mission Report: {time.strftime('%Y-%m-%d')}"]
    report_lines.append(f"\nTotal Landmarks Found: {len(app.state.landmarks)}\n---")

    for landmark in app.state.landmarks:
        report_lines.append(f"\n## Landmark Detail: {landmark.id}")
        report_lines.append(f"- **Name/Category:** {landmark.name}")
        report_lines.append(f"- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(landmark.timestamp))}")
        report_lines.append(f"- **Location (X, Y, Z):** ({landmark.location.x}, {landmark.location.y}, {landmark.location.z})")
        report_lines.append(f"- **Visual Description:**\n  > {landmark.detailed_description.replace('\n', '\n  > ')}")
        report_lines.append(f"- **Contextual Analysis:**\n  > {landmark.contextual_analysis.replace('\n', '\n  > ')}")
        report_lines.append("\n---")
    
    return "\n".join(report_lines)
