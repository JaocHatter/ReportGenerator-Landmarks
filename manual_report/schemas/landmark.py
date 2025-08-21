from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Position(BaseModel):
    x: float
    y: float
    z: float

class Landmark(BaseModel):
    id: int
    name: str
    location: Position
    timestamp: datetime
