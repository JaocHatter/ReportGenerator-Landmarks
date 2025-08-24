from pydantic import BaseModel, Field
import time
from typing import Optional

class Position(BaseModel):
    """
    Represents a 3D position of the rover or a landmark.
    """
    x: float = Field(..., description="X-coordinate in meters")
    y: float = Field(..., description="Y-coordinate in meters")
    z: float = Field(..., description="Z-coordinate in meters")

class LandmarkMetadata(BaseModel):
    """
    Represents the metadata sent alongside the landmark image during the request.
    """
    position: Position
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the image was captured.")

class Landmark(BaseModel):
    """
    Represents a fully analyzed and confirmed landmark record.
    This is the main data model stored in our application state.
    """
    id: str = Field(..., description="Unique identifier for the landmark (e.g., LM_1724298858_a3b1).")
    name: str = Field(..., description="The name or category of the object, as identified by the model.")
    location: Position = Field(..., description="The estimated 3D position of the landmark.")
    timestamp: float = Field(..., description="Unix timestamp of the observation.")
    best_image_path: Optional[str] = Field(None, description="File path to the best image of the landmark.")
    detailed_description: Optional[str] = Field(None, description="A detailed visual description from the model.")
    contextual_analysis: Optional[str] = Field(None, description="Contextual analysis regarding the object's origin, utility, and importance.")
