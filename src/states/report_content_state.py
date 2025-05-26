from typing import TypedDict, List, Any, Optional

class GeneralFindingsContent(TypedDict):
    """
    Contenido para la sección de hallazgos generales del reporte.
    """
    mission_id: str
    map_image_path: str # Path a la imagen del mapa generado
    llm_summary: Optional[str]
    total_landmarks_found: int

class LandmarkPageContent(TypedDict):
    """
    Contenido para la página individual de cada landmark.
    """
    landmark_id: str
    landmark_photo_path: str
    textual_description_full: str # Incluye object_name, visual_description, contextual_analysis
    location_info_formatted: str # Texto descriptivo de la ubicación y/o coordenadas
    # any_other_data_formatted: Optional[str]

class ReportContentState(TypedDict):
    """
    Estado que contiene todo el contenido listo para ser compilado en PDF.
    """
    mission_id: str
    report_filename: str # Nombre del archivo PDF a generar
    general_findings: GeneralFindingsContent
    landmark_pages: List[LandmarkPageContent]