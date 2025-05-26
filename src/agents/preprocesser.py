# agents/preprocesser.py
import os
import subprocess
import json
from typing import List, Optional
from states import MissionInputState, RobotPose
from states.preprocessed_video_segment_state import PreprocessedVideoSegmentState

class PreprocessorAgent:
    def __init__(self, segment_output_dir: str = "output/temp_video_segments"):
        self.segment_output_dir = segment_output_dir
        if not os.path.exists(self.segment_output_dir):
            os.makedirs(self.segment_output_dir, exist_ok=True)
        self.SEGMENT_DURATION_SECONDS = 5 * 60

    def _get_video_duration_seconds(self, video_path: str) -> Optional[float]:
        """Obtiene la duración total del video en segundos usando ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return float(result.stdout.strip())
        except FileNotFoundError:
            print("Error: ffprobe no encontrado. Asegúrate de que FFmpeg (y ffprobe) esté instalado y en el PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error al obtener la duración del video con ffprobe: {e.stderr}")
            return None
        except ValueError as e:
            print(f"Error al parsear la duración del video: {e}. Output: {result.stdout}")
            return None

    def _get_poses_for_segment(
        self,
        segment_start_ms: int,
        segment_end_ms: int,
        all_poses: List[RobotPose]
    ) -> List[RobotPose]:
        """Filtra las poses que caen dentro del rango de tiempo del segmento."""
        return [
            pose for pose in all_poses
            if segment_start_ms <= pose['timestamp_ms'] < segment_end_ms
        ]

    def process_mission_video(self, mission_input: MissionInputState) -> List[PreprocessedVideoSegmentState]:
        """
        Prepara el video para el análisis con Gemini, segmentándolo en trozos de 5 minutos.
        """
        video_segments_for_analysis: List[PreprocessedVideoSegmentState] = []
        video_path = mission_input['video_path']
        mission_id = mission_input['mission_id']

        if not os.path.exists(video_path):
            print(f"Error: El archivo de video no existe en {video_path}")
            return []

        total_duration_seconds = self._get_video_duration_seconds(video_path)
        if total_duration_seconds is None:
            print(f"No se pudo determinar la duración del video {video_path}. No se puede segmentar.")
            # Como fallback, podrías procesar el video completo como antes, o fallar.
            # Por ahora, fallaremos la segmentación.
            return []
        
        if total_duration_seconds <= 0:
            print(f"La duración del video {video_path} es 0 o inválida. No se puede segmentar.")
            return []

        print(f"Duración total del video '{video_path}': {total_duration_seconds:.2f} segundos.")

        num_segments = int(total_duration_seconds // self.SEGMENT_DURATION_SECONDS) + \
                       (1 if total_duration_seconds % self.SEGMENT_DURATION_SECONDS > 0 else 0)

        if num_segments == 0 and total_duration_seconds > 0: # Video muy corto pero válido
            num_segments = 1
            
        print(f"Dividiendo el video en {num_segments} segmento(s) de ~{self.SEGMENT_DURATION_SECONDS / 60} minutos.")

        for i in range(num_segments):
            segment_start_seconds = i * self.SEGMENT_DURATION_SECONDS
            
            # Determinar la duración real de este segmento
            # (puede ser menor que SEGMENT_DURATION_SECONDS para el último segmento)
            current_segment_duration = min(self.SEGMENT_DURATION_SECONDS, total_duration_seconds - segment_start_seconds)

            if current_segment_duration <= 0: # Evitar segmentos de duración cero si algo salió mal
                continue

            segment_filename = f"{mission_id}_segment_{i+1:03d}_{os.path.basename(video_path)}"
            output_segment_path = os.path.join(self.segment_output_dir, segment_filename)

            print(f"Procesando segmento {i+1}/{num_segments}: de {segment_start_seconds}s a {segment_start_seconds + current_segment_duration}s")
            print(f"Output para segmento: {output_segment_path}")

            # Comando ffmpeg para extraer el segmento
            # Usamos -c copy si es posible para evitar re-encodeo y hacerlo más rápido,
            # pero el usuario especificó -c:v libx264 -preset ultrafast, lo que implica re-encodeo.
            # Si el formato original es compatible y solo se quiere cortar, -c copy sería ideal.
            # Por ahora, seguiré la especificación del usuario.
            cmd = [
                "ffmpeg",
                "-y",  # Sobrescribir archivo de salida si existe
                "-i", video_path,
                "-ss", str(segment_start_seconds),
                "-t", str(current_segment_duration),
                "-c:v", "libx264", # Codec de video especificado
                "-preset", "ultrafast", # Preset para velocidad de encodeo
                "-an", # Opcional: remover audio si no es necesario para el análisis visual
                output_segment_path
            ]

            try:
                # Crear el directorio de salida si no existe (por si acaso, aunque el init lo hace)
                os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Segmento {i+1} creado exitosamente: {output_segment_path}")

                segment_start_ms = int(segment_start_seconds * 1000)
                segment_end_ms = int((segment_start_seconds + current_segment_duration) * 1000)

                poses_for_this_segment = self._get_poses_for_segment(
                    segment_start_ms,
                    segment_end_ms,
                    mission_input['robot_poses']
                )

                video_segments_for_analysis.append(PreprocessedVideoSegmentState(
                    mission_id=mission_id,
                    video_segment_path=output_segment_path,
                    start_time_in_original_video_ms=segment_start_ms,
                    end_time_in_original_video_ms=segment_end_ms,
                    robot_poses_for_segment=poses_for_this_segment
                ))

            except FileNotFoundError:
                print("Error: ffmpeg no encontrado. Asegúrate de que FFmpeg esté instalado y en el PATH.")
                # Podrías decidir detener todo el proceso aquí o continuar sin este segmento.
                return [] # Fallar si ffmpeg no está
            except subprocess.CalledProcessError as e:
                print(f"Error al crear el segmento {i+1} con ffmpeg:")
                print(f"Comando: {' '.join(e.cmd)}")
                print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
                # Continuar con el siguiente segmento o detenerse? Por ahora, continuamos si es posible.
                # Pero si un segmento falla, puede ser problemático.
            except Exception as e_gen:
                print(f"Error inesperado al procesar el segmento {i+1}: {e_gen}")


        if not video_segments_for_analysis and total_duration_seconds > 0:
             print(f"Advertencia: No se generaron segmentos de video, pero el video original tenía duración. Verifique los logs de ffmpeg.")
        elif video_segments_for_analysis:
            print(f"Agente Preprocesador: Video '{video_path}' segmentado en {len(video_segments_for_analysis)} parte(s).")
        
        return video_segments_for_analysis

    def run(self, mission_input: MissionInputState) -> List[PreprocessedVideoSegmentState]:
        print(f"Agente Preprocesador (Segmentación con FFmpeg): Iniciando para misión {mission_input['mission_id']}...")
        return self.process_mission_video(mission_input)