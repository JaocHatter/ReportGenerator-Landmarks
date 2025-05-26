# agents/identifier.py
import os
import uuid
from typing import List, Dict, Tuple
from states import (
    ConfirmedLandmarkState,
    IdentifiedLandmarksBatchState,
    RobotPose
)
from states.analyzed_video_segment_state import AnalyzedVideoSegmentState
from utils.gemini_client import get_gemini_model, generate_text 
import cv2  

class IdentifierAgent:
    def __init__(self, output_landmark_image_dir: str = "output/landmark_images"):
        self.gemini_model = get_gemini_model()
        self.output_landmark_image_dir = output_landmark_image_dir
        if not os.path.exists(self.output_landmark_image_dir):
            os.makedirs(self.output_landmark_image_dir, exist_ok=True)

        if not self.gemini_model:
            print("IdentifierAgent: ADVERTENCIA - Modelo Gemini no inicializado. El análisis contextual no funcionará.")

    def _extract_specific_frame(self, video_path: str, timestamp_ms: int, output_image_path: str) -> bool:
        """
        Extrae un fotograma de un video en un timestamp específico y lo guarda.
        Retorna True si tiene éxito, False en caso contrario.
        """
        if not os.path.exists(video_path):
            print(f"Error _extract_specific_frame: El archivo de video no existe en {video_path}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error _extract_specific_frame: No se pudo abrir el video {video_path}")
            return False
        
        # Asegurarse que el timestamp no sea negativo
        if timestamp_ms < 0:
            print(f"Advertencia _extract_specific_frame: Timestamp negativo ({timestamp_ms}ms) para video {video_path}. Usando 0ms.")
            timestamp_ms = 0

        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            try:
                cv2.imwrite(output_image_path, frame)
                # print(f"Fotograma para landmark guardado en: {output_image_path}") # Log opcional
                return True
            except Exception as e:
                print(f"Error _extract_specific_frame: Al guardar fotograma extraído en {output_image_path}: {e}")
                return False
        else:
            # Intentar con el primer frame si el timestamp falla y es > 0
            if timestamp_ms > 0:
                print(f"Advertencia _extract_specific_frame: No se pudo leer el fotograma en {timestamp_ms}ms del video {video_path}. Intentando con el primer fotograma.")
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    ret_fallback, frame_fallback = cap.read()
                    cap.release()
                    if ret_fallback and frame_fallback is not None:
                        try:
                            cv2.imwrite(output_image_path, frame_fallback)
                            print(f"Fotograma de fallback (inicio) para landmark guardado en: {output_image_path}")
                            return True
                        except Exception as e_fallback:
                            print(f"Error _extract_specific_frame: Al guardar fotograma de fallback: {e_fallback}")
                            return False
            print(f"Error _extract_specific_frame: No se pudo leer ningún fotograma útil del video {video_path} para el timestamp {timestamp_ms}ms.")
            return False

    def _build_contextual_analysis_prompt(self, landmark_description: str, category_suggestion: str) -> str:
        """
        Construye el prompt para el análisis contextual del landmark por Gemini.
        """
        return f"""
Eres un experto en misiones de exploración de Marte (ERC 2025).
Se ha identificado un objeto en Marte con la siguiente descripción y categoría sugerida:
Categoría Sugerida: {category_suggestion if category_suggestion else "No especificada"}
Descripción Visual Observada: "{landmark_description}"

Por favor, proporciona un análisis contextual conciso para el reporte final, siguiendo este formato estrictamente:
NOMBRE_OBJETO: [Nombre o categoría refinada del objeto. Sé específico si es posible. Ej: "Taladro geológico de mano", "Panel de control de módulo científico", "Fragmento de escudo térmico".]
DESCRIPCION_DETALLADA: [Elabora brevemente sobre la descripción visual observada, añadiendo detalles inferidos si es lógico. Si la descripción ya es buena, puedes refinarla o resumirla.]
ANALISIS_CONTEXTUAL: [
    - Origen probable: ¿Es natural de Marte? Si no, ¿podría ser de una misión anterior, actual, o es completamente anómalo?
    - Utilidad potencial: ¿Podría este objeto ser útil para la misión actual del rover, para futuras misiones o una base? ¿Cómo?
    - Relevancia/Importancia: ¿Qué tan significativo es este hallazgo en el contexto de la exploración de Marte?
    - Peligros/Consideraciones: ¿Presenta algún peligro obvio o consideración especial?
]
"""

    def _parse_contextual_response(self, response_text: str, default_description: str) -> Tuple[str, str, str]:
        """
        Parsea la respuesta del análisis contextual de Gemini.
        Retorna (object_name, detailed_description, contextual_analysis_text).
        """
        obj_name = default_description[:70] # Un default razonable
        det_desc = default_description
        ctx_analysis = "Análisis contextual no disponible o no parseable."

        # Eliminar bloques de código markdown si existen
        clean_text = response_text.strip()
        if clean_text.startswith("```") and clean_text.endswith("```"):
            clean_text = clean_text[3:-3].strip()
            if clean_text.lower().startswith("json"): 
                 clean_text = clean_text[len("json"):].strip()


        current_section = None
        temp_det_desc = []
        temp_ctx_analysis = []

        for line in clean_text.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith("NOMBRE_OBJETO:"):
                obj_name = line_stripped.split("NOMBRE_OBJETO:", 1)[1].strip()
                current_section = "name"
            elif line_stripped.startswith("DESCRIPCION_DETALLADA:"):
                det_desc_part = line_stripped.split("DESCRIPCION_DETALLADA:", 1)[1].strip()
                if det_desc_part: temp_det_desc.append(det_desc_part)
                current_section = "description"
            elif line_stripped.startswith("ANALISIS_CONTEXTUAL:"):
                ctx_analysis_part = line_stripped.split("ANALISIS_CONTEXTUAL:", 1)[1].strip()
                if ctx_analysis_part: temp_ctx_analysis.append(ctx_analysis_part)
                current_section = "context"
            elif current_section == "description" and line_stripped:
                temp_det_desc.append(line_stripped)
            elif current_section == "context" and line_stripped:
                temp_ctx_analysis.append(line_stripped)
        
        if temp_det_desc:
            det_desc = "\n".join(temp_det_desc)
        if temp_ctx_analysis:
            ctx_analysis = "\n".join(temp_ctx_analysis)
            
        return obj_name, det_desc, ctx_analysis


    def _find_closest_robot_pose(self, target_timestamp_ms: int, all_poses: List[RobotPose]) -> RobotPose:
        """
        Encuentra la pose del robot más cercana a un timestamp global dado.
        """
        if not all_poses:
            # Retorna una pose por defecto si no hay datos, o podrías lanzar un error.
            return RobotPose(timestamp_ms=target_timestamp_ms, x=0.0, y=0.0, orientation_degrees=0.0)
        
        closest_pose = min(all_poses, key=lambda pose: abs(pose['timestamp_ms'] - target_timestamp_ms))
        return closest_pose

    def run(
        self,
        analyzed_segments_batch: List[AnalyzedVideoSegmentState],
        full_robot_path_poses: List[RobotPose]
    ) -> IdentifiedLandmarksBatchState:
        """
        Procesa los segmentos de video analizados, identifica y contextualiza landmarks.
        """
        print(f"Agente Identificador: Iniciando. Procesando {len(analyzed_segments_batch)} segmento(s) de video analizado(s).")
        confirmed_landmarks_list: List[ConfirmedLandmarkState] = []
        
        if not analyzed_segments_batch:
            print("Agente Identificador: No hay segmentos analizados para procesar.")
            return IdentifiedLandmarksBatchState(
                mission_id="unknown_mission_no_segments",
                confirmed_landmarks=[],
                full_robot_path_poses=full_robot_path_poses
            )

        # Asumimos que todos los segmentos son de la misma misión
        mission_id = analyzed_segments_batch[0].processed_segment_info['mission_id']
        
        landmark_counter = 0

        for analyzed_segment in analyzed_segments_batch:
            segment_info = analyzed_segment.processed_segment_info
            original_video_segment_path = segment_info['video_segment_path']
            # El timestamp de inicio del segmento actual dentro del video completo de la misión
            segment_start_time_in_mission_ms = segment_info['start_time_in_original_video_ms']

            if not analyzed_segment.identified_landmark_observations:
                # print(f"Agente Identificador: Segmento {segment_info['video_segment_path']} no contiene observaciones de landmarks.")
                continue

            for obs in analyzed_segment.identified_landmark_observations:
                landmark_counter += 1
                landmark_id_str = f"LM_{mission_id}_{landmark_counter:03d}"
                
                # Path para la imagen del landmark que se extraerá
                best_image_filename = f"{mission_id}_{landmark_id_str}.jpg"
                best_image_filepath = os.path.join(self.output_landmark_image_dir, best_image_filename)

                # Extraer el fotograma de mejor visibilidad del video/segmento original
                extraction_success = self._extract_specific_frame(
                    video_path=original_video_segment_path,
                    timestamp_ms=obs['best_visibility_timestamp_in_segment_ms'], # Timestamp relativo al segmento
                    output_image_path=best_image_filepath
                )
                
                image_path_for_report = best_image_filepath if extraction_success else "path/to/default/no_image_extracted.jpg" # Placeholder

                # Calcular el timestamp global en la misión para la ubicación del landmark
                landmark_timestamp_in_mission_ms = segment_start_time_in_mission_ms + obs['best_visibility_timestamp_in_segment_ms']
                
                # Encontrar la pose del robot más cercana a este timestamp global
                estimated_lm_pose = self._find_closest_robot_pose(
                    target_timestamp_ms=landmark_timestamp_in_mission_ms,
                    all_poses=full_robot_path_poses
                )
                
                # Análisis contextual con Gemini
                # Usar la descripción de la observación como base para el prompt contextual.
                # 'object_category' podría ser un campo que Gemini llene en la etapa de Analyst, si se le pide.
                object_category_suggestion = obs.get('object_category', "Objeto no categorizado") 
                
                contextual_prompt = self._build_contextual_analysis_prompt(
                    landmark_description=obs['object_description'],
                    category_suggestion=object_category_suggestion
                )
                
                contextual_response_text = f"Análisis contextual para '{obs['object_description']}' no disponible (Modelo Gemini no inicializado o error en la API)."
                if self.gemini_model:
                    print(f"Agente Identificador: Solicitando análisis contextual para Landmark '{landmark_id_str}' ('{obs['object_description'][:30]}...').")
                    contextual_response_text = generate_text(self.gemini_model, contextual_prompt)
                
                # Parsear la respuesta contextual
                obj_name, det_desc, ctx_analysis = self._parse_contextual_response(
                    contextual_response_text, 
                    obs['object_description']
                )

                confirmed_lm = ConfirmedLandmarkState(
                    landmark_id=landmark_id_str,
                    mission_id=mission_id,
                    best_image_path=image_path_for_report,
                    object_name_or_category=obj_name,
                    detailed_visual_description=det_desc,
                    contextual_analysis=ctx_analysis,
                    estimated_location=estimated_lm_pose,
                    frames_observed_timestamps=[landmark_timestamp_in_mission_ms], # Timestamp global de mejor visibilidad
                )
                confirmed_landmarks_list.append(confirmed_lm)
                print(f"Agente Identificador: Landmark '{landmark_id_str}' procesado. Nombre: '{obj_name}'.")

        if not confirmed_landmarks_list:
            print(f"Agente Identificador: No se confirmaron landmarks para la misión {mission_id} después de procesar todos los segmentos.")
        else:
            print(f"Agente Identificador: {len(confirmed_landmarks_list)} landmark(s) confirmado(s) en total para la misión {mission_id}.")
            
        return IdentifiedLandmarksBatchState(
            mission_id=mission_id,
            confirmed_landmarks=confirmed_landmarks_list,
            full_robot_path_poses=full_robot_path_poses 
        )