import subprocess
import os
import argparse

def extract_subclip(
    input_video_path: str,
    output_video_path: str,
    start_time: str, # Formato: "HH:MM:SS" o segundos como string/float
    duration: str,   # Formato: "HH:MM:SS" o segundos como string/float
    re_encode: bool = False # Si es True, re-codifica. Si es False, intenta copiar streams (más rápido).
):
    """
    Extrae un subclip de un video usando ffmpeg.

    Args:
        input_video_path (str): Ruta al video de entrada.
        output_video_path (str): Ruta donde se guardará el subclip.
        start_time (str): Tiempo de inicio del subclip (ej: "00:01:30" o "90").
        duration (str): Duración del subclip desde el start_time (ej: "00:00:10" o "10").
        re_encode (bool): Si es True, usa libx264 para re-codificar (cortes más precisos pero más lento).
                          Si es False (default), usa '-c copy' (más rápido, sin pérdida de calidad,
                          pero los cortes pueden ser menos precisos si no caen en keyframes).
    Returns:
        bool: True si la extracción fue exitosa, False en caso contrario.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: El archivo de video de entrada no existe: {input_video_path}")
        return False

    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directorio de salida creado: {output_dir}")

    # Comando base de ffmpeg
    cmd = [
        "ffmpeg",
        "-y",  # Sobrescribir archivo de salida si existe
        "-i", input_video_path,
        "-ss", str(start_time), # Tiempo de inicio
        "-t", str(duration),    # Duración del clip
    ]

    if re_encode:
        # Opciones para re-codificar (cortes más precisos, pero más lento)
        # Puedes ajustar los parámetros de libx264 según tus necesidades
        cmd.extend([
            "-c:v", "libx264",   # Codec de video
            "-preset", "medium", # Preset de velocidad/calidad (ultrafast, fast, medium, slow, etc.)
            "-crf", "23",        # Factor de calidad (0-51, menor es mejor calidad, 18-28 es un rango común)
            "-c:a", "aac",       # Codec de audio
            "-strict", "experimental", # Necesario para algunos builds de ffmpeg con aac
            "-b:a", "128k"       # Bitrate de audio
        ])
    else:
        # Opciones para copiar streams (rápido, sin pérdida de calidad, cortes en keyframes)
        cmd.extend([
            "-c", "copy", # Copia los codecs de audio y video
            "-avoid_negative_ts", "make_zero" # Evita timestamps negativos que pueden causar problemas
        ])

    cmd.append(output_video_path)

    print(f"Ejecutando comando ffmpeg: {' '.join(cmd)}")

    try:
        process = subprocess.run(
            cmd,
            check=True, # Lanza CalledProcessError si ffmpeg retorna un código de error
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True # Decodifica stdout y stderr como texto
        )
        print(f"Subclip generado exitosamente: {output_video_path}")
        # print(f"ffmpeg stdout:\n{process.stdout}") # Descomentar para ver salida detallada
        if process.stderr: # A veces ffmpeg usa stderr para información, no solo errores
             print(f"ffmpeg stderr (puede contener información útil o advertencias):\n{process.stderr}")
        return True
    except FileNotFoundError:
        print("Error: ffmpeg no encontrado. Asegúrate de que FFmpeg esté instalado y en el PATH de tu sistema.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error durante la ejecución de ffmpeg:")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Código de retorno: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e_gen:
        print(f"Ocurrió un error inesperado al generar el subclip: {e_gen}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae un subclip de un video MP4 usando ffmpeg.")
    parser.add_argument("input_video", help="Ruta al archivo de video MP4 de entrada.")
    parser.add_argument("output_video", help="Ruta donde se guardará el subclip MP4.")
    parser.add_argument("start_time", help="Tiempo de inicio del subclip (ej: '00:01:30' para 1 minuto 30 segundos, o '90' para 90 segundos).")
    parser.add_argument("duration", help="Duración del subclip desde el start_time (ej: '00:00:10' para 10 segundos, o '10' para 10 segundos).")
    parser.add_argument(
        "--re_encode",
        action="store_true", # Si se pasa esta bandera, re_encode será True
        help="Re-codifica el video para cortes más precisos (más lento). Por defecto, copia los streams (más rápido)."
    )

    args = parser.parse_args()

    # --- Ejemplo de cómo podrías usarlo si quisieras enviar la primera mitad de un video ---
    # Primero, necesitarías obtener la duración total del video original.
    # (Esto requeriría ffprobe, como en ejemplos anteriores, o alguna otra librería)
    # Aquí, asumimos que el usuario provee el inicio y la duración directamente.

    # Ejemplo: extraer 10 segundos de video a partir del segundo 5
    # python tu_script.py video_original.mp4 subclip_video.mp4 5 10

    # Ejemplo: extraer desde el minuto 1, durante 30 segundos, y re-codificar
    # python tu_script.py video_original.mp4 subclip_video.mp4 00:01:00 30 --re_encode
    
    print(f"Input: {args.input_video}")
    print(f"Output: {args.output_video}")
    print(f"Start: {args.start_time}")
    print(f"Duration: {args.duration}")
    print(f"Re-encode: {args.re_encode}")

    if extract_subclip(args.input_video, args.output_video, args.start_time, args.duration, args.re_encode):
        print("Proceso completado.")
    else:
        print("El proceso falló.")