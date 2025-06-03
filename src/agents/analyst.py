# agents/analyst.py
import uuid
from typing import List
# Importar los nuevos estados
from states.preprocessed_video_segment_state import PreprocessedVideoSegmentState
from states.analyzed_video_segment_state import AnalyzedVideoSegmentState, LandmarkObservation
# Importar la nueva función del cliente Gemini
from utils.gemini_client import get_gemini_model, generate_analysis_from_video_file

class AnalystAgent:
    def __init__(self):
        self.gemini_model = get_gemini_model()
        if not self.gemini_model:
            print("AnalystAgent: ADVERTENCIA - Modelo Gemini no inicializado.")

    def _build_prompt_for_video_analysis(self, segment_info: PreprocessedVideoSegmentState) -> str:
        prompt = f"""
Eres un sistema avanzado de análisis de video para un rover en Marte (ERC 2025).
Analiza el siguiente VIDEO COMPLETO (o segmento de video) tomado por el rover.
El segmento de video cubre desde {segment_info['start_time_in_original_video_ms']}ms hasta {segment_info['end_time_in_original_video_ms']}ms del video original de la misión.
Los datos de pose del robot para este segmento han sido registrados.
Tu tarea es:
    Revisa TODO el video. Identifica CUIDADOSAMENTE cualquier objeto que NO parezca ser terreno natural marciano (rocas, arena, polvo, colinas distantes, cielo).
    Busca específicamente:
        - Objetos hechos por humanos o artificiales.
        - Herramientas, equipos, contenedores, infraestructura.
        - Objetos con colores muy distintivos del entorno (colores brillantes, metálicos no oxidados).
        - Objetos con formas geométricas regulares o complejas que no sean naturales.
        -Cualquier cosa que consideres un 'Landmark' potencial según las reglas del ERC (objetos físicos, no características geológicas primarias).

    IMPORTANTE: IGNORA LOS ARTEFACTOS VISUALES DE LA CÁMARA. No consideres como Landmarks los siguientes fenómenos, ya que son producto de la cámara o la transmisión y no objetos reales en el entorno:
        - Distorsiones de lente (efecto "ojo de pez" en los bordes, curvaturas inusuales).
        - "Digital video artifacts" o artefactos de compresión (bloques, pixelación excesiva).
        - Líneas horizontales o verticales de colores puros o patrones de interferencia que claramente no son parte de un objeto físico.
        - Destellos de lente (lens flares) o reflejos internos de la óptica.
        - Manchas en la lente o polvo que parezcan estar "flotando" o fijas en la imagen independientemente del movimiento del rover.
    Para cada OBJETO POTENCIALMENTE NO MARCIANO (candidato a Landmark) que identifiques en el video (y que NO sea un artefacto de cámara):
        - Proporciona una breve descripción del objeto.
        - Explica por qué crees que podría ser un Landmark (distintividad visual, forma, color, comportamiento temporal, y que no es un artefacto de la cámara).
        - Indica el timestamp de inicio (en milisegundos, relativo AL INICIO DE ESTE VIDEO/SEGMENTO) donde el objeto se vuelve visible por primera vez o es identificable.
        - Indica el timestamp de fin (en milisegundos, relativo AL INICIO DE ESTE VIDEO/SEGMENTO) donde el objeto deja de ser visible o relevante.
        - Indica el timestamp de mejor visibilidad (en milisegundos, relativo AL INICIO DE ESTE VIDEO/SEGMENTO) donde el objeto se ve más claro o es más fácil de identificar.
        - Comenta sobre su estabilidad (ej. "estático durante toda la observación", "se mueve lentamente").
Formato de salida esperado para CADA candidato a landmark (repite este bloque para cada uno):
LANDMARK_OBSERVATION_START
CANDIDATE_ID: [un ID único corto para esta observación, ej: LM_OBS_XYZ]
OBJECT_DESCRIPTION: [descripción]
REASONING_FOR_CANDIDACY: [por qué es candidato, incluyendo estabilidad y apariencia, y confirmación de que no es un artefacto de cámara]
START_TIMESTAMP_MS: [timestamp_inicio_ms]
END_TIMESTAMP_MS: [timestamp_fin_ms]
BEST_VISIBILITY_TIMESTAMP_MS: [timestamp_mejor_visibilidad_ms]
LANDMARK_OBSERVATION_END

Si no hay candidatos claros en todo el video/segmento (que no sean artefactos de cámara), indica "No se encontraron Landmarks potenciales en este segmento."
"""
        return prompt

    def _parse_gemini_video_response(self, response_text: str) -> List[LandmarkObservation]:
        observations: List[LandmarkObservation] = []
        clean_response_text = response_text.strip()

        parts = clean_response_text.split("LANDMARK_OBSERVATION_START")
        for part_idx, part in enumerate(parts[1:]):
            obs_data_str = part.split("LANDMARK_OBSERVATION_END")[0].strip()
            
            cand_id = f"lm_obs_{part_idx}_{uuid.uuid4().hex[:4]}"
            desc = "N/A"
            reasoning = "N/A"
            start_ts, end_ts, best_ts = 0, 0, 0

            for line in obs_data_str.split('\n'):
                if line.startswith("CANDIDATE_ID:"):
                    cand_id = line.split("CANDIDATE_ID:", 1)[1].strip()
                elif line.startswith("OBJECT_DESCRIPTION:"):
                    desc = line.split("OBJECT_DESCRIPTION:", 1)[1].strip()
                elif line.startswith("REASONING_FOR_CANDIDACY:"):
                    reasoning = line.split("REASONING_FOR_CANDIDACY:", 1)[1].strip()
                elif line.startswith("START_TIMESTAMP_MS:"):
                    try: start_ts = int(line.split("START_TIMESTAMP_MS:", 1)[1].strip())
                    except ValueError: start_ts = 0
                elif line.startswith("END_TIMESTAMP_MS:"):
                    try: end_ts = int(line.split("END_TIMESTAMP_MS:", 1)[1].strip())
                    except ValueError: end_ts = 0
                elif line.startswith("BEST_VISIBILITY_TIMESTAMP_MS:"):
                    try: best_ts = int(line.split("BEST_VISIBILITY_TIMESTAMP_MS:", 1)[1].strip())
                    except ValueError: best_ts = 0
            
            observations.append(LandmarkObservation(
                landmark_candidate_id=cand_id,
                object_description=desc,
                reasoning_for_candidacy=reasoning,
                start_timestamp_in_segment_ms=start_ts,
                end_timestamp_in_segment_ms=end_ts,
                best_visibility_timestamp_in_segment_ms=best_ts
            ))
        return observations

    def analyze_video_segment(self, segment_state: PreprocessedVideoSegmentState) -> AnalyzedVideoSegmentState:
        if not self.gemini_model:
            return AnalyzedVideoSegmentState(
                processed_segment_info=segment_state,
                gemini_full_video_analysis_text="Error: Modelo Gemini no disponible.",
                identified_landmark_observations=[]
            )

        prompt = self._build_prompt_for_video_analysis(segment_state)
        
        print(f"AnalystAgent: Analizando segmento de video {segment_state['video_segment_path']} para misión {segment_state['mission_id']}...")
        
        gemini_response_text = generate_analysis_from_video_file(
            model=self.gemini_model,
            prompt=prompt,
            video_file_path=segment_state['video_segment_path']
        )
        
        landmark_observations = self._parse_gemini_video_response(gemini_response_text)
        
        return AnalyzedVideoSegmentState(
            processed_segment_info=segment_state,
            gemini_full_video_analysis_text=gemini_response_text,
            identified_landmark_observations=landmark_observations
        )

    def run(self, video_segments: List[PreprocessedVideoSegmentState]) -> List[AnalyzedVideoSegmentState]:
        print(f"Agente Analista (Video Nativo): Iniciando análisis de {len(video_segments)} segmento(s) de video...")
        analyzed_segments_list: List[AnalyzedVideoSegmentState] = []
        for segment_state in video_segments:
            analyzed_segments_list.append(self.analyze_video_segment(segment_state))
        print(f"Agente Analista (Video Nativo): Análisis completado.")
        return analyzed_segments_list