import google.genai as genai
import os
import time

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
    exit()

client = None
try:
    # Using an alias to avoid conflict if needed, and for clarity with SDK examples
    client = genai.Client(api_key=GOOGLE_API_KEY)

    print("INFO: Cliente de Gemini inicializado.")
except Exception as e:
    print(f"Error al inicializar el cliente de Gemini: {e}")
    exit()

video_file_path = "/home/jaoc/Desktop/Projects/ERC/videos/primera_parte.mp4"
uploaded_file_resource = None  # Initialize

if not os.path.exists(video_file_path):
    print(f"Error: El archivo de video no existe en {video_file_path}")
    exit()

try:
    print(f"INFO: Subiendo archivo de video: {video_file_path}...")
    uploaded_file_resource = client.files.upload(file=video_file_path)
    print(f"INFO: Video subido. Nombre del recurso: {uploaded_file_resource.name}, Estado inicial: {uploaded_file_resource.state.name}")

    # === WAITING LOOP ===
    while uploaded_file_resource.state.name == "PROCESSING":
        print(f"INFO: El video '{uploaded_file_resource.name}' aún se está procesando. Esperando 10 segundos...")
        time.sleep(10)  # Wait for 10 seconds
        print(uploaded_file_resource.name)
        uploaded_file_resource = client.files.get(name=uploaded_file_resource.name)
        print(f"INFO: Estado actualizado del video '{uploaded_file_resource.name}': {uploaded_file_resource.state.name}")

    if uploaded_file_resource.state.name == "FAILED":
        print(f"ERROR: El procesamiento del video '{uploaded_file_resource.name}' por Gemini falló.")
        if uploaded_file_resource and uploaded_file_resource.name:
            try:
                client.files.delete(name=uploaded_file_resource.name)
                print(f"INFO: Archivo fallido '{uploaded_file_resource.name}' borrado de Gemini.")
            except Exception as e_del_failed:
                print(f"WARN: No se pudo borrar el archivo fallido '{uploaded_file_resource.name}': {e_del_failed}")
        exit()
    
    if uploaded_file_resource.state.name != "ACTIVE":
        print(f"ERROR: El video '{uploaded_file_resource.name}' no está en estado ACTIVO después del procesamiento. Estado final: {uploaded_file_resource.state.name}")
        if uploaded_file_resource and uploaded_file_resource.name:
            try:
                client.files.delete(name=uploaded_file_resource.name)
                print(f"INFO: Archivo '{uploaded_file_resource.name}' con estado {uploaded_file_resource.state.name} borrado de Gemini.")
            except Exception as e_del_non_active:
                print(f"WARN: No se pudo borrar el archivo '{uploaded_file_resource.name}': {e_del_non_active}")
        exit()

    print(f"INFO: Video '{uploaded_file_resource.name}' está ACTIVO. Generando contenido para la primera mitad...")

    model_name_for_call = "models/gemini-2.5-flash-preview-05-20" # Your specified model

    response = client.models.generate_content(
        model=model_name_for_call,
        contents=[
            uploaded_file_resource, 
            "Summarize this video. Then create a quiz with an answer key based on the information in this video."
        ]
    )

    print("\n--- Respuesta de Gemini (primera mitad del video) ---")
    print(response.text)

except Exception as e:
    print(f"Ocurrió un error general: {e}")
    import traceback
    traceback.print_exc()

finally:
    if uploaded_file_resource and uploaded_file_resource.name and client:
        try:
            print(f"INFO: Intentando borrar el archivo '{uploaded_file_resource.name}' de Gemini...")
            client.files.delete(name=uploaded_file_resource.name)
            print(f"INFO: Archivo '{uploaded_file_resource.name}' borrado de Gemini exitosamente.")
        except Exception as e_del:
            print(f"WARN: No se pudo borrar el archivo '{uploaded_file_resource.name}' de Gemini después de su uso: {e_del}")