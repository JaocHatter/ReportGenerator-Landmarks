from typing import TypedDict, List, Optional
from .mission_input_state import RobotPose

class PreprocessedVideoSegmentState(TypedDict):
    """
    Representa un segmento de video (o el video completo) listo para análisis,
    con su path y las poses asociadas.
    """
    mission_id: str
    video_segment_path: str # Path al archivo de video (o segmento)
    # Opcional: video_segment_gcs_uri: str # Si se sube a GCS para Gemini
    start_time_in_original_video_ms: int # Para rastrear si es un segmento
    end_time_in_original_video_ms: int   # Para rastrear si es un segmento
    robot_poses_for_segment: List[RobotPose] # Poses relevantes para este segmento