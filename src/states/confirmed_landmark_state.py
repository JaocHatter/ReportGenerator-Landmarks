from typing import TypedDict, List, Any, Optional
from .mission_input_state import RobotPose

class ConfirmedLandmarkState(TypedDict):
    """
    Representa un Landmark confirmado con toda la información necesaria para el reporte.
    """
    landmark_id: str
    mission_id: str
    best_image_path: str # Path a la imagen guardada del landmark
    
    object_name_or_category: str # Ej: "Caja de herramientas roja", "Panel solar dañado"
    detailed_visual_description: str
    contextual_analysis: str # Relación con Marte, utilidad, probabilidad, etc.
    
    estimated_location: RobotPose # Pose del robot (o coord. derivadas) cuando se vio mejor
    frames_observed_timestamps: List[int]

class IdentifiedLandmarksBatchState(TypedDict):
    """
    Contiene la lista de todos los landmarks confirmados y la ruta del robot.
    """
    mission_id: str
    confirmed_landmarks: List[ConfirmedLandmarkState]
    full_robot_path_poses: List[RobotPose]