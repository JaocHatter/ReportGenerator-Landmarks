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
from google.genai import types
import cv2  
import asyncio

class IdentifierAgent:
    def __init__(self, output_landmark_image_dir: str = "output/landmark_images"):
        self.gemini_model = get_gemini_model()
        self.output_landmark_image_dir = output_landmark_image_dir
        if not os.path.exists(self.output_landmark_image_dir):
            os.makedirs(self.output_landmark_image_dir, exist_ok=True)

        if not self.gemini_model:
            print("IdentifierAgent: WARNING - Gemini not initialized. Contextual Analysis won't work.")

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
                return True
            except Exception as e:
                print(f"Error _extract_specific_frame: Al guardar fotograma extraído en {output_image_path}: {e}")
                return False
        else:
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

    def _build_contextual_analysis_prompt(self) -> str:
        """
        Constructs the prompt for the contextual analysis of the landmark by Gemini.
        """
        return f"""
        You are an expert in Mars exploration missions.
        An object has been identified on Mars. An image of this object will be provided.

        Please provide a concise contextual analysis for the final report, STRICTLY following this format (No markdown):
        OBJECT_NAME: [Name or refined category of the object. Be specific if possible. E.g., "Handheld geological drill," "Scientific module control panel," "Heat shield fragment."]
        DETAILED_DESCRIPTION: [Briefly elaborate on the observed visual description, adding inferred details if logical. If the description is already good, you can refine or summarize it.]
        CONTEXTUAL_ANALYSIS: [
            - Probable origin: Is it natural to Mars? If not, could it be from a previous or current mission, or is it completely anomalous?
            - Potential utility: Could this object be useful for the current rover mission, for future missions, or a base? How?
            - Relevance/Importance: How significant is this finding in the context of Mars exploration?
            - Dangers/Considerations: Does it present any obvious dangers or special considerations?
        ]
        """

    def _parse_contextual_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        Parsea la respuesta del análisis contextual de Gemini.
        Retorna (object_name, detailed_description, contextual_analysis_text).
        """
        ctx_analysis = "Contextual Analysis not available"

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
            if line_stripped.startswith("OBJECT_NAME:"):
                obj_name = line_stripped.split("OBJECT_NAME:", 1)[1].strip()
                current_section = "name"
            elif line_stripped.startswith("DETAILED_DESCRIPTION:"):
                det_desc_part = line_stripped.split("DETAILED_DESCRIPTION:", 1)[1].strip()
                if det_desc_part: temp_det_desc.append(det_desc_part)
                current_section = "description"
            elif line_stripped.startswith("CONTEXTUAL_ANALYSIS:"):
                ctx_analysis_part = line_stripped.split("CONTEXTUAL_ANALYSIS:", 1)[1].strip()
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
            return RobotPose(timestamp_ms=target_timestamp_ms, x=0.0, y=0.0, orientation_degrees=0.0)
        
        closest_pose = min(all_poses, key=lambda pose: abs(pose['timestamp_ms'] - target_timestamp_ms))
        return closest_pose
    
    async def identify_responses(self, segments_prompt_image: List[tuple]) -> List[types.GenerateContentResponse]:
        api_tasks = [self.gemini_model.generate_content_from_image(data[0],data[1]) for data in segments_prompt_image]
        responses = await asyncio.gather(*api_tasks)
        return responses
    
    async def run(
        self,
        analyzed_segments_batch: List[AnalyzedVideoSegmentState],
        full_robot_path_poses: List[RobotPose]
    ) -> IdentifiedLandmarksBatchState:
        """
        Procesa los segmentos de video analizados, identifica y contextualiza landmarks.
        """
        print(f"Identifier Agent: STARTING. Processing {len(analyzed_segments_batch)} segment(s) from analyzed video.")
        confirmed_landmarks_list: List[ConfirmedLandmarkState] = []
        
        if not analyzed_segments_batch:
            print("Agente Identificador: No hay segmentos analizados para procesar.")
            return IdentifiedLandmarksBatchState(
                mission_id="unknown_mission_no_segments",
                confirmed_landmarks=[],
                full_robot_path_poses=full_robot_path_poses
            )

        mission_id = analyzed_segments_batch[0]["processed_segment_info"]['mission_id']

        landmark_counter = 0
        contextual_prompt = self._build_contextual_analysis_prompt()
        segments_prompt_image = []
        obs_landmarks = []

        for analyzed_segment in analyzed_segments_batch:
            segment_info = analyzed_segment["processed_segment_info"]
            original_video_segment_path = segment_info['video_segment_path']
            # El timestamp de inicio del segmento actual dentro del video completo de la misión
            segment_start_time_in_mission_ms = segment_info['start_time_in_original_video_ms']

            if not analyzed_segment["identified_landmark_observations"]:
                # print(f"Agente Identificador: Segmento {segment_info['video_segment_path']} no contiene observaciones de landmarks.")
                continue

            for obs in analyzed_segment["identified_landmark_observations"]:
                landmark_counter += 1
                landmark_id_str = f"LM_{mission_id}_{landmark_counter:03d}"
                
                # Path para la imagen del landmark que se extraerá
                best_image_filename = f"{mission_id}_{landmark_id_str}.jpg"
                best_image_filepath = os.path.join(self.output_landmark_image_dir, best_image_filename)

                # Extraer el fotograma de mejor visibilidad del video/segmento original
                extraction_success = self._extract_specific_frame(
                    video_path=original_video_segment_path,
                    timestamp_ms=obs['best_visibility_timestamp_in_segment_ms'],
                    output_image_path=best_image_filepath
                )
                
                image_path_for_report = best_image_filepath if extraction_success else "path/to/default/no_image_extracted.jpg" 

                # Calcular el timestamp global en la misión para la ubicación del landmark
                landmark_timestamp_in_mission_ms = segment_start_time_in_mission_ms + obs['best_visibility_timestamp_in_segment_ms']
                
                # Encontrar la pose del robot más cercana a este timestamp global
                estimated_lm_pose = self._find_closest_robot_pose(
                    target_timestamp_ms=landmark_timestamp_in_mission_ms,
                    all_poses=full_robot_path_poses
                )

                contextual_response_text = f"Contextual Analysis not available."
                if self.gemini_model:
                    with open(image_path_for_report, 'rb') as f:
                        image_bytes = f.read()
                        print(f"⏳ Identifier Agent: Solicitude to get a contextual analysis of the landamark: '{landmark_id_str}'")
                        segments_prompt_image.append((contextual_prompt, image_bytes))
                        obs_landmarks.append(obs)
                
        responses = await self.identify_responses(segments_prompt_image)

        for response, obs in zip(responses, obs_landmarks):
            # Parsear la respuesta contextual
            obj_name, det_desc, ctx_analysis = self._parse_contextual_response(response.text)

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