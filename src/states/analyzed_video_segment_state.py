from typing import TypedDict, List, Optional
from .preprocessed_video_segment_state import PreprocessedVideoSegmentState 

class LandmarkObservation(TypedDict):
    """
    Información detallada sobre un Landmark observado en el video/segmento.
    """
    landmark_name: str 
    start_timestamp_in_segment_ms: int
    end_timestamp_in_segment_ms: int
    best_visibility_timestamp_in_segment_ms: int

class AnalyzedVideoSegmentState(TypedDict):
    """
    Estado después del análisis de un segmento de video por Gemini.
    """
    processed_segment_info: PreprocessedVideoSegmentState
    gemini_full_video_analysis_text: str 
    identified_landmark_observations: List[LandmarkObservation]