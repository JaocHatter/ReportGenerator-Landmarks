# agents/preprocesser.py
import os
import subprocess
import json
import cv2
from typing import List, Optional
from states import MissionInputState, RobotPose
from states.preprocessed_video_segment_state import PreprocessedVideoSegmentState

class PreprocessorAgent:
    def __init__(self, segment_output_dir: str = "output/temp_video_segments"):
        self.segment_output_dir = segment_output_dir
        if not os.path.exists(self.segment_output_dir):
            os.makedirs(self.segment_output_dir, exist_ok=True)
        self.SEGMENT_DURATION_SECONDS = 300 # 5 min

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
            print("Error: ffprobe not found. Make sure FFmpeg (and ffprobe) is installed and in your PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error getting video duration with ffprobe: {e.stderr}")
            return None
        except ValueError as e:
            print(f"Error parsing the video duration: {e}. Output: {result.stdout}")
            return None

    def _write_timestamp_on_video(self, input_path: str, output_path: str) -> bool:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video de entrada en {input_path}")
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Error: No se pudo crear el archivo de video de salida en {output_path}")
            cap.release()
            return False

        print(f"✍️  Escribiendo timestamps en el video. Fuente: {input_path}")
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

        print("✅ Proceso de escritura de timestamps completado.")
        cap.release()
        writer.release()
        return True

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
        video_segments_for_analysis: List[PreprocessedVideoSegmentState] = []
        video_path = mission_input['video_path']
        mission_id = mission_input['mission_id']

        if not os.path.exists(video_path):
            print(f"Error: The video file does not exist at {video_path}")
            return []
        
        timestamped_video_filename = f"{mission_id}_with_timestamp.mp4"
        timestamped_video_path = os.path.join(self.segment_output_dir, timestamped_video_filename)
        
        if not self._write_timestamp_on_video(video_path, timestamped_video_path):
            print("Error: Could not write timestamps on the video. Segmentation cannot proceed.")
            return []

        video_path_for_segmentation = timestamped_video_path
        total_duration_seconds = self._get_video_duration_seconds(video_path_for_segmentation)
        if total_duration_seconds is None:
            print(f"Could not determine the duration of the video {video_path_for_segmentation}. Segmentation cannot proceed.")
            # As a fallback, you could process the full video as before, or fail.
            # For now, we'll fail the segmentation.
            return []

        if total_duration_seconds <= 0:
            print(f"The duration of the video {video_path_for_segmentation} is 0 or invalid. Segmentation cannot proceed.")
            return []

        print(f"Total duration of the video '{video_path_for_segmentation}': {total_duration_seconds:.2f} seconds.")

        num_segments = int(total_duration_seconds // self.SEGMENT_DURATION_SECONDS) + \
                       (1 if total_duration_seconds % self.SEGMENT_DURATION_SECONDS > 0 else 0)

        if num_segments == 0 and total_duration_seconds > 0: # Video muy corto pero válido
            num_segments = 1
            
        print(f"Splitting the video into {num_segments} segment(s) of ~{self.SEGMENT_DURATION_SECONDS / 60} minutes.")

        for i in range(num_segments):
            segment_start_seconds = i * self.SEGMENT_DURATION_SECONDS
            
            current_segment_duration = min(self.SEGMENT_DURATION_SECONDS, total_duration_seconds - segment_start_seconds)

            if current_segment_duration <= 0: # Evitar segmentos de duración cero si algo salió mal
                continue

            segment_filename = f"{mission_id}_segment_{i+1:03d}_{os.path.basename(video_path_for_segmentation)}"
            output_segment_path = os.path.join(self.segment_output_dir, segment_filename)

            print(f"Processing {i+1}/{num_segments} segment: from {segment_start_seconds}s to {segment_start_seconds + current_segment_duration}s")
            print(f"Segment Output: {output_segment_path}")

            # Comando ffmpeg para extraer el segmento
            # Usamos -c copy si es posible para evitar re-encodeo y hacerlo más rápido,
            # pero el usuario especificó -c:v libx264 -preset ultrafast, lo que implica re-encodeo.
            # Si el formato original es compatible y solo se quiere cortar, -c copy sería ideal.
            # Por ahora, seguiré la especificación del usuario.
            cmd = [
                "ffmpeg",
                "-y",  # Sobrescribir archivo de salida si existe
                "-i", video_path_for_segmentation,
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
                print(f"Segment {i+1} created sucessfully: {output_segment_path}")

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
                print("Error: ffmpeg not found. Make sure FFmpeg is installed and in the PATH.")
                # You could choose to stop the entire process here or continue without this segment.
                return []  # Fail if ffmpeg is not found
            except subprocess.CalledProcessError as e:
                print(f"Error while creating segment {i+1} with ffmpeg:")
                print(f"Command: {' '.join(e.cmd)}")
                print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
            except Exception as e_gen:
                print(f"Unexpected error while processing segment {i+1}: {e_gen}")

        if not video_segments_for_analysis and total_duration_seconds > 0:
            print(f"Warning: No video segments were generated, but the original video had duration. Check the ffmpeg logs.")
        elif video_segments_for_analysis:
            print(f"Preprocessor Agent: Video '{video_path_for_segmentation}' segmented into {len(video_segments_for_analysis)} part(s).")

        return video_segments_for_analysis

    def run(self, mission_input: MissionInputState) -> List[PreprocessedVideoSegmentState]:
        print(f"Preprocessor Agent (Segmentation with FFmpeg): Starting for mission {mission_input['mission_id']}...")
        return self.process_mission_video(mission_input)