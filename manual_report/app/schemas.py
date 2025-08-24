from pydantic import BaseModel, Field
import time
from typing import Optional

class Position(BaseModel):
    """
    Representa una posición 3D del rover o un landmark.
    """
    x: float = Field(..., description="Coordenada X en metros")
    y: float = Field(..., description="Coordenada Y en metros")
    z: float = Field(..., description="Coordenada Z en metros")

class LandmarkMetadata(BaseModel):
    """
    Representa los metadatos enviados junto con la imagen del landmark durante la solicitud.
    """
    position: Position
    timestamp: float = Field(default_factory=time.time, description="Timestamp Unix de cuando se capturó la imagen.")

class Landmark(BaseModel):
    """
    Representa un registro de landmark completamente analizado y confirmado.
    Este es el modelo de datos principal almacenado en el estado de nuestra aplicación.
    """
    id: str = Field(..., description="Identificador único para el landmark (ej., LM_1724298858_a3b1).")
    name: str = Field(..., description="El nombre o categoría del objeto, según lo identificado por el modelo.")
    location: Position = Field(..., description="La posición 3D estimada del landmark.")
    timestamp: float = Field(..., description="Timestamp Unix de la observación.")
    best_image_path: Optional[str] = Field(None, description="Ruta del archivo a la mejor imagen del landmark.")
    detailed_description: Optional[str] = Field(None, description="Una descripción visual detallada del modelo.")
    contextual_analysis: Optional[str] = Field(None, description="Análisis contextual sobre el origen, utilidad e importancia del objeto.")

class Orientation(BaseModel):
    """Define la orientación del rover."""
    roll: float
    pitch: float
    yaw: float

class Pose(BaseModel):
    """Combina la posición y orientación del rover."""
    position: Position
    orientation: Orientation

class PoseData(BaseModel):
    """El modelo raíz para los datos de pose recibidos."""
    pose: Pose