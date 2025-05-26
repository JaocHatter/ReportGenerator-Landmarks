from typing import TypedDict, List, Optional

class RobotPose(TypedDict):
    """
    Representa la pose del robot en un momento dado.
    """
    timestamp_ms: int
    x: float
    y: float
    orientation_degrees: float
    # Opcional: gps_latitude: Optional[float]
    # Opcional: gps_longitude: Optional[float]

class MissionInputState(TypedDict):
    """
    Estado inicial con la ruta del video y los datos de pose del robot.
    """
    video_path: str
    robot_poses: List[RobotPose] # Lista de poses o path a un archivo de poses
    mission_id: str # Identificador de la misión/corrida