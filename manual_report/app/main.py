from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
import os
import logging
import json
import shutil
import time
import hashlib
from contextlib import asynccontextmanager
from app.schemas import Landmark, LandmarkMetadata, PoseData, Position
from app.report_generator import ReportGenerator
from app.gemini_service import GeminiService

# --- Configuración de directorios y logging ---
OUTPUT_DIR = "output"
LANDMARKS_DATA_DIR = os.path.join(OUTPUT_DIR, "landmarks_data")
TRAJECTORY_DATA_DIR = os.path.join(OUTPUT_DIR, "trajectory_data")
LANDMARK_IMAGES_DIR = os.path.join(OUTPUT_DIR, "landmark_images")
MARKS_FILE = os.path.join(LANDMARKS_DATA_DIR, "markers.json")
TRAJECTORY_FILE = os.path.join(TRAJECTORY_DATA_DIR, "path.txt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lifespan para gestionar el ciclo de vida de la aplicación ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando la aplicación...")
    os.makedirs(LANDMARKS_DATA_DIR, exist_ok=True)
    os.makedirs(TRAJECTORY_DATA_DIR, exist_ok=True)
    os.makedirs(LANDMARK_IMAGES_DIR, exist_ok=True)
    
    if not os.path.exists(MARKS_FILE):
        with open(MARKS_FILE, "w") as f: json.dump([], f)
    if not os.path.exists(TRAJECTORY_FILE):
        with open(TRAJECTORY_FILE, "w") as f: f.write("")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("La variable de entorno GOOGLE_API_KEY no está configurada.")
        app.state.gemini_service = None
    else:
        app.state.gemini_service = GeminiService(api_key=api_key)
        logger.info("GeminiService inicializado correctamente.")
    
    yield
    logger.info("Apagando la aplicación...")

app = FastAPI(lifespan=lifespan)

# --- Endpoints ---

@app.post("/add_landmark/", response_model=Landmark, status_code=201)
async def add_landmark(
    request: Request,
    metadata_json: str = Form(..., description="Un string JSON que se valida con el schema LandmarkMetadata."),
    image: UploadFile = File(...)
):
    """Recibe datos de un landmark, los analiza y los almacena."""
    try:
        metadata = LandmarkMetadata.model_validate_json(metadata_json)
        image_bytes = await image.read()
        
        gemini_service: GeminiService = request.app.state.gemini_service
        if not gemini_service:
            raise HTTPException(status_code=503, detail="El servicio de análisis de imágenes no está disponible.")
            
        analysis_result = await gemini_service.get_contextual_analysis(image_bytes)

        if not analysis_result or not analysis_result.get("object_name"):
            raise HTTPException(status_code=400, detail="No se pudo identificar un nombre de landmark válido.")

        landmark_id = f"LM_{int(time.time())}"
        image_filename = f"{landmark_id}{os.path.splitext(image.filename)[1] or '.png'}"
        image_filepath = os.path.join(LANDMARK_IMAGES_DIR, image_filename)

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

        landmarks_list = []
        if os.path.exists(MARKS_FILE):
            with open(MARKS_FILE, "r") as f:
                landmarks_list = json.load(f)
        
        landmarks_list.append(new_landmark.model_dump())

        with open(MARKS_FILE, "w") as f:
            json.dump(landmarks_list, f, indent=2)
        
        logger.info(f"Nuevo landmark añadido: {new_landmark.name} (ID: {new_landmark.id})")
        
        return new_landmark

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error inesperado en add_landmark: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.post("/add_pose/", status_code=201)
async def add_pose(data: PoseData):
    pos = data.pose.position
    try:
        with open(TRAJECTORY_FILE, "a") as f:
            f.write(f"{pos.x},{pos.y}\n")
        return JSONResponse(content={"status": "success", "message": "Pose guardada."})
    except Exception as e:
        logger.error(f"Fallo al escribir en el archivo de trayectoria: {e}")
        raise HTTPException(status_code=500, detail="Fallo al guardar la trayectoria.")

@app.post("/generate_report/", response_model=dict)
async def generate_and_save_report(
    pgm_file: UploadFile = File(...),
    yaml_file: UploadFile = File(...)
):
    """
    Genera un informe en PDF usando los archivos de mapa y los datos guardados.
    """
    temp_pgm_path = os.path.join(OUTPUT_DIR, f"temp_{pgm_file.filename}")
    temp_yaml_path = os.path.join(OUTPUT_DIR, f"temp_{yaml_file.filename}")
    
    try:
        # Guardar los archivos de mapa subidos temporalmente
        with open(temp_pgm_path, "wb") as buffer: shutil.copyfileobj(pgm_file.file, buffer)
        with open(temp_yaml_path, "wb") as buffer: shutil.copyfileobj(yaml_file.file, buffer)

        landmarks_list = []
        if os.path.exists(MARKS_FILE):
            with open(MARKS_FILE, 'r') as f: 
                landmarks_list = json.load(f)

        if not landmarks_list:
            raise HTTPException(status_code=404, detail="No hay landmarks para generar un informe.")

        # Preparar los datos para el ReportGenerator
        map_files = {
            'pgm': temp_pgm_path, 
            'yaml': temp_yaml_path,
            'trajectory': TRAJECTORY_FILE
        }

        # La llamada al constructor ahora es correcta
        report_generator = ReportGenerator(
            landmarks_data=landmarks_list, 
            map_files=map_files
        )
        pdf_filepath = report_generator.generate_report()
        
        if not os.path.exists(pdf_filepath):
            raise HTTPException(status_code=500, detail="El archivo del informe no fue creado.")

        return {
            "status": "success",
            "message": "Informe generado.",
            "filepath": pdf_filepath
        }
    except Exception as e:
        logger.error(f"Fallo al generar el informe: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # La limpieza de los archivos temporales se maneja ahora dentro del ReportGenerator
        pass
