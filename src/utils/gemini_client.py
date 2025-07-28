import google.genai as genai, types
import os
from typing import List, Dict, Any 
import time
import pathlib
import google.genai as genai
from google.genai import types

# TODO
# Add asynchronism

gemini_client_instance = None
INITIALIZATION_ERROR_MESSAGE = None

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    
    # Use genai.Client as shown in the provided documentation context
    gemini_client_instance = genai.Client(api_key=GOOGLE_API_KEY)
    print("INFO: Cliente de Gemini inicializado correctamente.")

except ValueError as e:
    INITIALIZATION_ERROR_MESSAGE = f"Error de configuración de Gemini: {e}"
    print(INITIALIZATION_ERROR_MESSAGE)
except Exception as e_global_init:
    INITIALIZATION_ERROR_MESSAGE = f"Error inesperado durante la inicialización global de Gemini: {e_global_init}"
    print(INITIALIZATION_ERROR_MESSAGE)


# User-specified model name - DO NOT CHANGE
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

class ModelExecutionWrapper:
    """
    A wrapper to allow using client-based API while maintaining
    a model.generate_content(...) call signature within other functions,
    as per the constraint of not changing function arguments.
    """
    def __init__(self, client: genai.Client, model_name_str: str):
        if not client:
            raise ValueError("Se requiere una instancia de genai.Client para ModelExecutionWrapper.")
        if not model_name_str:
            raise ValueError("Se requiere un nombre de modelo para ModelExecutionWrapper.")
            
        self.client = client
        if not model_name_str.startswith("models/"):
            self.model_to_call = f"models/{model_name_str}"
        else:
            self.model_to_call = model_name_str
        print(f"INFO: ModelExecutionWrapper inicializado para el modelo: {self.model_to_call}")

    def generate_content(self, contents: Any) -> types.GenerateContentResponse:
        """
        Wraps the client.generate_content call.
        'contents' can be a string (for text prompts) or a list (for multimodal prompts).
        """
        if not self.client:
            raise RuntimeError("Gemini client not available in ModelExecutionWrapper.")
        
        return self.client.models.generate_content(model=self.model_to_call, contents=contents)
    
    async def generate_content_from_video(self, prompt: Any, video_bytes: bytes) -> types.GenerateContentResponse:
        response = await self.client.aio.models.generate_content(
            model = "gemini-2.5-flash",
            contents = [
                    types.Part(
                        inline_data = types.Blob(
                            data = video_bytes,
                            mime_type = 'video/mp4'   
                            ),
                        video_metadata = types.VideoMetadata(fps=5)
                        )
                    ,
                    types.Part(text = prompt)
            ],
            config = types.GenerateContentConfig(
                temperature = 0.5  
                )
        )
        return response 
    
    async def generate_content_from_image(self, prompt: str, image_bytes: bytes) -> types.GenerateContentResponse:
        response = await self.client.aio.models.generate_content(
            model = "gemini-2.5-flash",
            contents = [
                    types.Part(
                        inline_data = types.Blob(
                            data = image_bytes,
                            mime_type = 'image/jpeg'   
                            )
                        ),
                    types.Part(text = prompt)
            ],
            config = types.GenerateContentConfig(
                temperature = 0.5  
                )
        )
        return response 

def get_gemini_model() -> ModelExecutionWrapper | None:
    """
    Retorna una instancia del wrapper que permite llamadas tipo model.generate_content().
    Adheres to "No cambies el nombre de la funciones, ni sus argumentos".
    The return type hint genai.GenerativeModel is not strictly true anymore,
    but the returned object will have the required .generate_content method.
    """
    global gemini_client_instance, INITIALIZATION_ERROR_MESSAGE
    if not gemini_client_instance:
        # Log the original error if client initialization failed
        print(f"ERROR: No se puede obtener el 'modelo' porque el cliente de Gemini no se inicializó: {INITIALIZATION_ERROR_MESSAGE}")
        return None
    try:
        # Returns the wrapper instead of a direct genai.GenerativeModel instance
        return ModelExecutionWrapper(client=gemini_client_instance, model_name_str=MODEL_NAME)
    except Exception as e:
        print(f"No se pudo inicializar ModelExecutionWrapper para el modelo '{MODEL_NAME}': {e}")
        return None

def generate_text(model: ModelExecutionWrapper, prompt: str) -> str:
    """
    Genera texto a partir de un prompt usando el wrapper del modelo Gemini.
    Adheres to "No cambies el nombre de la funciones, ni sus argumentos".
    """
    if not model: 
        return "Error: Instancia de ModelExecutionWrapper no inicializada o inválida."
    try:
        response = model.generate_content(prompt) 
        return response.text
    except AttributeError as ae:
        if "text" not in str(ae) and hasattr(response, "candidates"): 
            try:
                return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            except Exception:
                pass 
        print(f"AttributeError durante la llamada a Gemini (texto): {ae}. Respuesta: {response if 'response' in locals() else 'No response object'}")
        return f"Error de atributo en Gemini (texto): {str(ae)}"
    except Exception as e:
        print(f"Error durante la llamada a Gemini (texto): {e}")
        return f"Error en Gemini (texto): {str(e)}"

def generate_analysis_from_video_file(
    model: ModelExecutionWrapper,
    prompt: str,
    video_file_path: str
) -> str:
    """
    Genera análisis a partir de un archivo de video y un prompt usando Gemini.
    Sube el archivo de video usando el cliente de Gemini.
    Adheres to "No cambies el nombre de la funciones, ni sus argumentos".
    """
    global gemini_client_instance 

    if not model:
        return "Error: Instancia de ModelExecutionWrapper no inicializada o inválida."
    if not gemini_client_instance:
        return f"Error: Cliente de Gemini no inicializado para operaciones de archivo. {INITIALIZATION_ERROR_MESSAGE}"
    
    if not os.path.exists(video_file_path):
        return f"Error: El archivo de video no existe en la ruta: {video_file_path}"

    print(f"INFO: Subiendo archivo de video: {video_file_path} a Gemini API...")
    uploaded_file_resource = None
    try:
        # Use client instance for file operations
        uploaded_file_resource = gemini_client_instance.files.upload(file=video_file_path)
        print(f"INFO: Video subido. Nombre del recurso: {uploaded_file_resource.name} (URI: {uploaded_file_resource.uri}), Estado: {uploaded_file_resource.state.name}")
    except Exception as e:
        print(f"ERROR: Al subir el video '{video_file_path}' a Gemini: {e}")
        return f"Error al subir el video a Gemini: {e}"

    try:
        # Esperar a que el video se procese
        while uploaded_file_resource.state.name == "PROCESSING":
            print(f"INFO: Esperando que Gemini procese el video '{uploaded_file_resource.name}'...")
            time.sleep(5) 
            uploaded_file_resource = gemini_client_instance.files.get(name=uploaded_file_resource.name)
            print(f"INFO: Estado actualizado del video '{uploaded_file_resource.name}': {uploaded_file_resource.state.name}")

        if uploaded_file_resource.state.name == "FAILED":
            error_msg = f"Error: El procesamiento del video por Gemini falló. Recurso: {uploaded_file_resource.name}, URI: {uploaded_file_resource.uri}"
            print(f"ERROR: {error_msg}")
            return error_msg
        
        if uploaded_file_resource.state.name != "ACTIVE":
            error_msg = f"Error: El video no alcanzó el estado 'ACTIVE'. Estado final: {uploaded_file_resource.state.name}. Recurso: {uploaded_file_resource.name}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
        print(f"INFO: Video '{uploaded_file_resource.name}' procesado y activo. Generando contenido...")
        response = model.generate_content([uploaded_file_resource, prompt]) 
        
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            full_text_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if full_text_response:
                return full_text_response
            else:
                print(f"WARN: Respuesta de Gemini recibida, pero no se encontró texto en las partes del candidato. Respuesta: {response}")
                return "Respuesta de Gemini recibida, pero sin contenido de texto extraíble."
        else:
            print(f"WARN: Respuesta de Gemini no tiene atributo 'text' ni candidatos con texto. Respuesta: {response}")
            return "Respuesta de Gemini en formato inesperado."

    except AttributeError as ae:
        print(f"AttributeError durante la llamada a Gemini (video): {ae}. Verifique la estructura de la respuesta.")
        return f"Error de atributo en Gemini (video): {str(ae)}"
    except Exception as e:
        print(f"Error durante la llamada a Gemini (video): {e}")
        return f"Error en Gemini (video): {str(e)}"
    finally:
        if uploaded_file_resource and uploaded_file_resource.name:
            try:
                print(f"INFO: Intentando borrar el archivo {uploaded_file_resource.name} de Gemini después de su uso...")
                gemini_client_instance.files.delete(name=uploaded_file_resource.name)
                print(f"INFO: Archivo {uploaded_file_resource.name} borrado de Gemini.")
            except Exception as e_del:
                print(f"WARN: No se pudo borrar el archivo {uploaded_file_resource.name} de Gemini: {e_del}")