from pydantic import BaseModel
from landmark import Landmark
from typing import List

class ReportMetadata(BaseModel):
    id: int
    landmarks: List[Landmark]