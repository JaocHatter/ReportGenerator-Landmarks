from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import os
from typing import List, Dict
import logging
from app.schemas import Landmark, Position, LandmarkMetadata, PoseData
from app.gemini_service import GeminiService
import time
from contextlib import asynccontextmanager
import json
import uuid

# --- CONFIGURACIÓN DE DIRECTORIOS Y ARCHIVOS ---
LANDMARK_IMAGE_DIR = "output/landmark_images"
TRAJECTORY_DATA_DIR = "output/trajectory_data"
TRAJECTORY_FILE = os.path.join(TRAJECTORY_DATA_DIR, "path.txt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación, inicializando servicios y creando directorios."""
    try:
        # Crear directorios necesarios al iniciar
        os.makedirs(LANDMARK_IMAGE_DIR, exist_ok=True)
        os.makedirs(TRAJECTORY_DATA_DIR, exist_ok=True)
        
        GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
        
        app.state.gemini_service = GeminiService(api_key=GEMINI_API_KEY)
        logger.info("Servicio Gemini inicializado correctamente.")

        # Almacenamiento temporal para landmarks
        app.state.landmarks = []
        
        yield
    except Exception as e:
        logger.critical(f"Error crítico durante el inicio de la aplicación: {e}")

app = FastAPI(lifespan=lifespan)

@app.post("/add_landmark/", response_model=Landmark, status_code=201)
async def add_landmark(
    image: UploadFile = File(..., description="Archivo de imagen del landmark potencial."),
    metadata_json: str = Form(..., description="Cadena JSON con metadatos del landmark (posición, timestamp).")
):
    """
    Recibe una imagen, la guarda, realiza un análisis contextual y almacena la información.
    """
    try:
        metadata = LandmarkMetadata(**json.loads(metadata_json))
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Formato JSON inválido para metadatos: {e}")

    try:
        image_bytes = await image.read()
        gemini_service: GeminiService = app.state.gemini_service
        analysis_result = await gemini_service.get_contextual_analysis(image_bytes)

        if not analysis_result or not analysis_result.get("object_name"):
            raise HTTPException(status_code=400, detail="No se pudo identificar un nombre de landmark válido en la imagen.")

        landmark_id = f"LM_{int(time.time())}"
        file_extension = os.path.splitext(image.filename)[1] or ".jpg"
        image_filename = f"{landmark_id}{file_extension}"
        image_filepath = os.path.join(LANDMARK_IMAGE_DIR, image_filename)

        with open(image_filepath, "wb") as buffer:
            buffer.write(image_bytes)
        
        absolute_image_path = os.path.abspath(image_filepath)

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
        logger.info(f"Nuevo landmark añadido: {new_landmark.name} (ID: {new_landmark.id})")
        return new_landmark

    except Exception as e:
        logger.error(f"Error inesperado en add_landmark: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error interno en el servidor.")

# --- NUEVO ENDPOINT PARA LA TRAYECTORIA ---
@app.post("/add_pose/", status_code=201)
async def add_pose(data: PoseData):
    """
    Recibe los datos de pose del rover y añade la posición al archivo de trayectoria.
    """
    try:
        pos = data.pose.position
        # 'a' significa modo 'append' (añadir al final)
        with open(TRAJECTORY_FILE, "a") as f:
            f.write(f"{pos.x},{pos.y}\n")
        logger.info(f"Posición añadida a la trayectoria: X={pos.x}, Y={pos.y}")
        return JSONResponse(content={"status": "success", "message": "Datos de pose añadidos."})
    except Exception as e:
        logger.error(f"Fallo al escribir los datos de pose: {e}")
        raise HTTPException(status_code=500, detail="Fallo al escribir los datos de pose.")

@app.get("/landmarks/", response_model=List[Landmark])
async def get_all_landmarks():
    return app.state.landmarks

@app.delete("/erase_last_landmark/", response_model=Landmark)
async def erase_last_landmark():
    if not app.state.landmarks:
        raise HTTPException(status_code=404, detail="No hay landmarks para borrar.")
    return app.state.landmarks.pop()
