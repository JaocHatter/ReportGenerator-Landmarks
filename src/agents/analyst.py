# agents/analyst.py
import uuid
from typing import List
from states.preprocessed_video_segment_state import PreprocessedVideoSegmentState
from states.analyzed_video_segment_state import AnalyzedVideoSegmentState, LandmarkObservation
from utils.gemini_client import get_gemini_model, generate_analysis_from_video_file
import asyncio
from google.genai import types

class AnalystAgent:
    def __init__(self):
        self.gemini_model = get_gemini_model()
        if not self.gemini_model:
            print("AnalystAgent: WARNING - Gemini Model not initialized.")

    def _build_prompt_for_video_analysis(self, segment_info: PreprocessedVideoSegmentState) -> str:
        prompt = f"""
        Analyze the provided video segment, which covers the interval from `{segment_info['start_time_in_original_video_ms']}`ms to `{segment_info['end_time_in_original_video_ms']}` of the original mission video.

        **Primary Objective:**
        Your task is to review the **ENTIRE** video segment and identify any objects that are clearly **NOT** natural Martian terrain (like rocks, sand, or dust).

        **Inclusion Criteria:**
        A candidate object **MUST MEET ALL** of the following conditions to be reported:

        1.  It is an **artificial or human-made object** or an inusual object in the martian terrain.
        2.  It is **fully and clearly visible** within the frame.
        3.  It appears **very close** to the camera.
        4.  It occupies a **significant portion** of the video frame.

        **Exclusion Criteria (CRITICAL):**
        **DO NOT REPORT CAMERA ARTIFACTS.** The following phenomena are products of the camera or transmission and must be **IGNORED**:

        * **Lens distortions:** "Fisheye" effects or unusual curvatures at the edges.
        * **Digital artifacts:** Compression blocks or excessive pixelation.
        * **Image interference:** Horizontal/vertical lines or patterns not part of a physical object.
        * **Optical effects:** Lens flares or internal reflections.
        * **Lens contamination:** Smudges or dust particles that are fixed or "float" on the camera lens.

        -----
        **Output Format:**
        For **EACH** valid landmark identified, use the following structured block. Be concise and direct.

        ```
        LANDMARK_OBSERVATION_START
        NAME: [A brief but descriptive name for the object]
        START_TIMESTAMP_MS: [start_timestamp_ms]
        END_TIMESTAMP_MS: [end_timestamp_ms]
        BEST_VISIBILITY_TIMESTAMP_MS: [The timestamp in milliseconds (ms) where the object best meets all inclusion criteria]
        LANDMARK_OBSERVATION_END
        ```

        If **NO** objects meeting **ALL** the above criteria are found in the entire segment, your **ONLY** response should be:
        `No significant landmarks found in this segment.`
        """
        return prompt

    def _parse_gemini_video_response(self, response_text: str) -> List[LandmarkObservation]:
        observations: List[LandmarkObservation] = []
        clean_response_text = response_text.strip()

        parts = clean_response_text.split("LANDMARK_OBSERVATION_START")
        for part in parts[1:]:
            obs_data_str = part.split("LANDMARK_OBSERVATION_END")[0].strip()
            
            start_ts, end_ts, best_ts = 0, 0, 0

            for line in obs_data_str.split('\n'):
                if line.startswith("NAME:"):
                    name = line.split("NAME:", 1)[1].strip()
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
                landmark_name=name,
                start_timestamp_in_segment_ms=start_ts,
                end_timestamp_in_segment_ms=end_ts,
                best_visibility_timestamp_in_segment_ms=best_ts
            ))
        return observations

    def analyze_video_segment(self, gemini_response_text: str ,segment_state: PreprocessedVideoSegmentState) -> AnalyzedVideoSegmentState:
        if not self.gemini_model:
            return AnalyzedVideoSegmentState(
                processed_segment_info=segment_state,
                gemini_full_video_analysis_text="Error: Gemini model not available",
                identified_landmark_observations=[]
            )

        print(f"Analyst Agent: Extracting data from {segment_state['video_segment_path']} for the mission {segment_state['mission_id']}...")
        
        landmark_observations = self._parse_gemini_video_response(gemini_response_text)
        
        return AnalyzedVideoSegmentState(
            processed_segment_info=segment_state,
            gemini_full_video_analysis_text=gemini_response_text,
            identified_landmark_observations=landmark_observations
        )
    
    async def analysis_responses(self, segments_prompt_video: List[tuple]) -> List[types.GenerateContentResponse]:        
        api_tasks = [self.gemini_model.generate_content_from_video(prompt= data[0], video_bytes= data[1]) for data in segments_prompt_video]
        results = await asyncio.gather(*api_tasks)
        return results

    async def run(self, video_segments: List[PreprocessedVideoSegmentState]) -> List[AnalyzedVideoSegmentState]:
        print(f"Analyst Agent: Starting analysis for {len(video_segments)} segment(s)...")
        analyzed_segments_list: List[AnalyzedVideoSegmentState] = []
        segments_prompt_video = [(self._build_prompt_for_video_analysis(segment_state), open(segment_state["video_segment_path"], 'rb').read()) \
                                 for segment_state in video_segments]
        
        # running asynchronously
        gemini_responses = await self.analysis_responses(segments_prompt_video)

        print(gemini_responses[0].text)

        for segment, response in zip(video_segments, gemini_responses):
            analyzed_segment = self.analyze_video_segment(
                gemini_response_text=response.text, 
                segment_state=segment
            )
            analyzed_segments_list.append(analyzed_segment)
        print(f"Analyst Agent: Analysis complete.")
        return analyzed_segments_list
    