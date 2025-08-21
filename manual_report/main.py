from fastapi import FastAPI
import os
import google.genai as genai
from google.genai import types
import logging
from schemas.landmark import Landmark
from contextlib import asynccontextmanager
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        MODEL_ID = "gemini-2.5-flash"
        if not GEMINI_API_KEY:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        
        app.state.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        app.state.gemini_model_id = MODEL_ID

        # temporal place to store landmarks
        app.state.landmarks = []
        app.state.landmarks_lock = asyncio.Lock()

        logger.info("Gemini client initialized successfully")
        yield
    except ValueError as e:
        INITIALIZATION_ERROR_MESSAGE = f"Error de configuración de Gemini: {e}"
        logger.error(INITIALIZATION_ERROR_MESSAGE)

app = FastAPI(lifespan=lifespan)

@app.post("/add_landmark")
async def add_landmark(image_bytes: bytes, metadata:dict, response_model: Landmark):
    prompt = """
    Hi, I'm a prompt
    """
    response = await app.state.gemini_client.aio.models.generate_content(
        model = app.state.gemini_model_id,
        content = [
            types.Part(
                inline_data = types.Blob(
                    data = image_bytes,
                    mime_type = 'image/jpeg'
                )
            ),
            types.Part(text = prompt)
        ],
        config = types.GenerateContentConfig(
            temperature = 0.2
        )
    )
    landmark = Landmark(
        id = response.id,
        name = response.name,
        location = metadata.get('position', None),
        timestamp = metadata.get('timestamp', 0)
    )

@app.get("/erase_last_landmark")
def erase_last_landmark():
    return

@app.get("/generate_report")
def generate_report():
    return

