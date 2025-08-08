import os
import subprocess
import cv2
import numpy as np
import json
from typing import TypedDict, List, Optional

# ==============================================================================
# 1. DEFINICIÓN DE ESTRUCTURAS DE DATOS (STATES)
# ==============================================================================

class RobotPose(TypedDict):
    """
    Representa la pose del robot en un momento dado.
    """
    timestamp_ms: int
    x: float
    y: float
    orientation_degrees: float

class MissionInputState(TypedDict):
    """
    Estado inicial con la ruta del video y los datos de pose del robot.
    """
    video_path: str
    robot_poses: List[RobotPose]
    mission_id: str

class PreprocessedVideoSegmentState(TypedDict):
    """
    Representa un segmento de video listo para análisis.
    """
    mission_id: str
    video_segment_path: str
    start_time_in_original_video_ms: int
    end_time_in_original_video_ms: int
    robot_poses_for_segment: List[RobotPose]

# ==============================================================================
# 2. CLASE DEL AGENTE PREPROCESADOR
# ==============================================================================

class PreprocessorAgent:
    def __init__(self, segment_output_dir: str = "output/temp_video_segments"):
        self.segment_output_dir = segment_output_dir
        if not os.path.exists(self.segment_output_dir):
            os.makedirs(self.segment_output_dir, exist_ok=True)
        # Para la demostración, usaremos segmentos más cortos de 15 segundos
        self.SEGMENT_DURATION_SECONDS = 15

    def _get_video_duration_seconds(self, video_path: str) -> Optional[float]:
        """Obtiene la duración total del video en segundos usando ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return float(result.stdout.strip())
        except FileNotFoundError:
            print("Error: ffprobe no encontrado. Asegúrate de que FFmpeg esté instalado y en el PATH.")
            return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Error al obtener la duración del video con ffprobe: {e}")
            return None

    def _write_timestamp_on_video(self, input_path: str, output_path: str) -> bool:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Was not possible open the video at {input_path}")
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Error: Was not possible create the output video file at {output_path}")
            cap.release()
            return False

        print(f"✍️  Writing timestamps en el video. Fuente: {input_path}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = int(timestamp_ms / 1000)
            ms = int(timestamp_ms % 1000)
            minutes = seconds // 60
            hours = minutes // 60
            timestamp_text = f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}.{ms:03d}"
            
            text_size = cv2.getTextSize(timestamp_text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = text_size[1] + 10
            
            cv2.putText(frame, timestamp_text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            writer.write(frame)

        print("✅ Timestamps written successfully.")
        cap.release()
        writer.release()
        return True

    def _get_poses_for_segment(self, segment_start_ms: int, segment_end_ms: int, all_poses: List[RobotPose]) -> List[RobotPose]:
        """Filtra las poses que caen dentro del rango de tiempo del segmento."""
        return [p for p in all_poses if segment_start_ms <= p['timestamp_ms'] < segment_end_ms]

    def process_mission_video(self, mission_input: MissionInputState) -> List[PreprocessedVideoSegmentState]:
        """Prepara el video para el análisis: añade timestamp y lo segmenta."""
        video_segments_for_analysis: List[PreprocessedVideoSegmentState] = []
        original_video_path = mission_input['video_path']
        mission_id = mission_input['mission_id']

        if not os.path.exists(original_video_path):
            print(f"Error: El archivo de video no existe en {original_video_path}")
            return []

        timestamped_video_filename = f"{mission_id}_with_timestamp.mp4"
        timestamped_video_path = os.path.join(self.segment_output_dir, timestamped_video_filename)
        
        if not self._write_timestamp_on_video(original_video_path, timestamped_video_path):
            print("No se pudo escribir el timestamp en el video. Abortando.")
            return []
        
        video_path_for_segmentation = timestamped_video_path
        total_duration_seconds = self._get_video_duration_seconds(video_path_for_segmentation)
        
        if not total_duration_seconds or total_duration_seconds <= 0:
            print(f"La duración del video {video_path_for_segmentation} es inválida.")
            return []

        print(f"⏱️  Duración total del video: {total_duration_seconds:.2f} segundos.")
        num_segments = int(np.ceil(total_duration_seconds / self.SEGMENT_DURATION_SECONDS))
        print(f"✂️  Dividiendo el video en {num_segments} segmento(s) de ~{self.SEGMENT_DURATION_SECONDS} segundos.")

        for i in range(num_segments):
            segment_start_seconds = i * self.SEGMENT_DURATION_SECONDS
            current_segment_duration = min(self.SEGMENT_DURATION_SECONDS, total_duration_seconds - segment_start_seconds)
            if current_segment_duration <= 0: continue

            segment_filename = f"{mission_id}_segment_{i+1:03d}.mp4"
            output_segment_path = os.path.join(self.segment_output_dir, segment_filename)

            print(f"\nProcessing segment {i+1}/{num_segments} -> {output_segment_path}")
            cmd = [
                "ffmpeg", "-y", "-i", video_path_for_segmentation,
                "-ss", str(segment_start_seconds), "-t", str(current_segment_duration),
                "-c:v", "libx264", "-preset", "ultrafast", "-an", output_segment_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Segmento {i+1} creado exitosamente.")

                segment_start_ms = int(segment_start_seconds * 1000)
                segment_end_ms = int((segment_start_seconds + current_segment_duration) * 1000)

                poses_for_this_segment = self._get_poses_for_segment(
                    segment_start_ms, segment_end_ms, mission_input['robot_poses']
                )

                video_segments_for_analysis.append(PreprocessedVideoSegmentState(
                    mission_id=mission_id,
                    video_segment_path=output_segment_path,
                    start_time_in_original_video_ms=segment_start_ms,
                    end_time_in_original_video_ms=segment_end_ms,
                    robot_poses_for_segment=poses_for_this_segment
                ))
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"Error al crear el segmento {i+1} con ffmpeg: {e}")
                return []

        print(f"\n🎉 Agente Preprocesador: Video '{original_video_path}' procesado en {len(video_segments_for_analysis)} parte(s).")
        return video_segments_for_analysis

    def run(self, mission_input: MissionInputState) -> List[PreprocessedVideoSegmentState]:
        print(f"🚀 Iniciando preprocesamiento para misión {mission_input['mission_id']}...")
        return self.process_mission_video(mission_input)

# ==============================================================================
# 3. FUNCIÓN PARA CREAR DATOS DE PRUEBA
# ==============================================================================

def create_dummy_video_and_poses(path: str, duration_sec: int, fps: int) -> List[RobotPose]:
    """Crea un video de prueba simple y una lista de poses de robot correspondientes."""
    print(f"🎬 Creando video de prueba: {path} ({duration_sec}s @ {fps}fps)")
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    
    poses = []
    num_frames = duration_sec * fps
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = f"Frame {i+1} / {num_frames}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame)
        
        # Generar una pose para este fotograma
        timestamp_ms = int((i / fps) * 1000)
        pose: RobotPose = {
            "timestamp_ms": timestamp_ms,
            "x": round(i * 0.1, 2),
            "y": round(np.sin(i / 10.0) * 5, 2),
            "orientation_degrees": (i % 360)
        }
        poses.append(pose)
        
    writer.release()
    print("✅ Video de prueba y poses creados.")
    return poses

# ==============================================================================
# 4. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # --- Configuración ---
    DUMMY_VIDEO_PATH = "dummy_video.mp4"
    OUTPUT_DIR = "output/mission_alpha"
    MISSION_ID = "mission_alpha"
    VIDEO_DURATION_SECONDS = 40 # Duración del video de prueba
    VIDEO_FPS = 30

    # --- Preparación ---
    # 1. Crear el video y las poses de prueba
    robot_poses_data = create_dummy_video_and_poses(
        DUMMY_VIDEO_PATH, 
        VIDEO_DURATION_SECONDS, 
        VIDEO_FPS
    )

    # 2. Preparar el estado de entrada para el agente
    mission_input: MissionInputState = {
        "video_path": DUMMY_VIDEO_PATH,
        "robot_poses": robot_poses_data,
        "mission_id": MISSION_ID
    }

    # --- Ejecución ---
    # 3. Instanciar y ejecutar el agente
    agent = PreprocessorAgent(segment_output_dir=OUTPUT_DIR)
    processed_segments = agent.run(mission_input)

    # --- Resultados ---
    # 4. Mostrar los resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS DEL PROCESAMIENTO")
    print("="*50)
    if processed_segments:
        for i, segment in enumerate(processed_segments):
            print(f"\n--- Segmento {i+1} ---")
            print(f"  ID Misión: {segment['mission_id']}")
            print(f"  Ruta archivo: {segment['video_segment_path']}")
            print(f"  Tiempo inicio (ms): {segment['start_time_in_original_video_ms']}")
            print(f"  Tiempo fin (ms): {segment['end_time_in_original_video_ms']}")
            print(f"  Poses asociadas: {len(segment['robot_poses_for_segment'])}")
            if segment['robot_poses_for_segment']:
                # Muestra la primera y última pose del segmento como ejemplo
                print(f"    - Primera pose: t={segment['robot_poses_for_segment'][0]['timestamp_ms']}ms")
                print(f"    - Última pose:  t={segment['robot_poses_for_segment'][-1]['timestamp_ms']}ms")
    else:
        print("No se generaron segmentos.")
    
    print("\n🧹 Para limpiar, elimina el archivo 'dummy_video.mp4' y la carpeta 'output'.")