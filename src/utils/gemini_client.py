import google.genai as genai
import os
from PIL import Image
import io
from typing import List, Dict, Any
import time

try:
    GOOGLE_API_KEY = os.getenv("AIzaSyCcTBlLvSZ8MqTlwIqve-NWsm8hVpH0ZwM")
    if not GOOGLE_API_KEY:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    genai.configure(api_key=GOOGLE_API_KEY)
except ValueError as e:
    print(f"Error de configuración de Gemini: {e}")

MODEL_NAME = "gemini-2.5-flash-preview-05-20"

def get_gemini_model():
    """Retorna una instancia del modelo Gemini."""
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"No se pudo inicializar el modelo Gemini: {e}")
        return None

def generate_text(model: genai.GenerativeModel, prompt: str) -> str:
    """
    Genera texto a partir de un prompt usando Gemini (solo texto).
    """
    if not model:
        return "Error: Modelo Gemini no inicializado."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error durante la llamada a Gemini (texto): {e}")
        return f"Error en Gemini: {str(e)}"

def generate_analysis_from_video_file(
    model: genai.GenerativeModel,
    prompt: str,
    video_file_path: str
) -> str:
    """
    Genera análisis a partir de un archivo de video y un prompt usando Gemini.
    Sube el archivo de video a la API de Gemini.
    """
    if not model:
        return "Error: Modelo Gemini no inicializado."
    
    print(f"Subiendo archivo de video: {video_file_path} a Gemini API...")
    try:
        video_file = genai.upload_file(path=video_file_path)
        print(f"Video subido: {video_file.name} ({video_file.uri})")
    except Exception as e:
        return f"Error al subir el video a Gemini: {e}"

    while video_file.state.name == "PROCESSING":
        print("Esperando que Gemini procese el video...")
        time.sleep(10) 
        video_file = genai.get_file(video_file.name) 

    if video_file.state.name == "FAILED":
        return f"Error: El procesamiento del video por Gemini falló. {video_file.uri}"
    
    print("Video procesado por Gemini. Generando contenido...")
    try:
        response = model.generate_content([prompt, video_file])
        return response.text
    except Exception as e:
        print(f"Error durante la llamada a Gemini (video): {e}")
        return f"Error en Gemini: {str(e)}"
    finally:
        try:
            print(f"Intentando borrar el archivo {video_file.name} de Gemini...")
            genai.delete_file(video_file.name)
            print(f"Archivo {video_file.name} borrado de Gemini.")
        except Exception as e_del:
            print(f"Advertencia: No se pudo borrar el archivo {video_file.name} de Gemini: {e_del}")