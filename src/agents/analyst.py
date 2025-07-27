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
        You are an advanced video analysis system for a rover on Mars (ERC 2025).
        Analyze the following COMPLETE VIDEO (or video segment) taken by the rover.
        The video segment covers from {segment_info['start_time_in_original_video_ms']}ms to {segment_info['end_time_in_original_video_ms']}ms of the original mission video.
        The robot's pose data for this segment has been recorded.
        Your task is:
            Review the ENTIRE video. CAREFULLY identify any objects that DO NOT appear to be natural Martian terrain (rocks, sand, dust, distant hills, sky).
            Specifically look for:
                - Human-made or artificial objects.
                - Tools, equipment, containers, infrastructure.
                - Objects with colors very distinct from the environment (bright colors, non-oxidized metallic).
                - Objects with regular or complex geometric shapes that are not natural.
                - Anything you consider a potential 'Landmark' according to ERC rules (physical objects, not primary geological features).

            IMPORTANT: IGNORE CAMERA VISUAL ARTIFACTS. Do not consider the following phenomena as Landmarks, as they are products of the camera or transmission and not real objects in the environment:
                - Lens distortions ("fisheye" effect at the edges, unusual curvatures).
                - "Digital video artifacts" or compression artifacts (blocks, excessive pixelation).
                - Horizontal or vertical lines of pure colors or interference patterns that are clearly not part of a physical object.
                - Lens flares or internal optical reflections.
                - Lens smudges or dust that appear to be "floating" or fixed in the image regardless of rover movement.
            For each POTENTIALLY NON-MARTIAN OBJECT (Landmark candidate) you identify in the video (and which is NOT a camera artifact):
                - Provide a brief description of the object.
                - Explain why you believe it could be a Landmark (visual distinctiveness, shape, color, temporal behavior, and confirmation that it is not a camera artifact).
                - Indicate the start timestamp (in milliseconds, relative TO THE START OF THIS VIDEO/SEGMENT) where the object first becomes visible or identifiable.
                - Indicate the end timestamp (in milliseconds, relative TO THE START OF THIS VIDEO/SEGMENT) where the object is no longer visible or relevant.
                - Indicate the best visibility timestamp (in milliseconds, relative TO THE START OF THIS VIDEO/SEGMENT) where the object is seen most clearly or is easiest to identify.
                - Comment on its stability (e.g., "static throughout the observation," "moves slowly").
        Expected output format for EACH landmark candidate (repeat this block for each one):
        LANDMARK_OBSERVATION_START
        CANDIDATE_ID: [a short unique ID for this observation, e.g., LM_OBS_XYZ]
        OBJECT_DESCRIPTION: [description]
        REASONING_FOR_CANDIDACY: [why it's a candidate, including stability and appearance, and confirmation that it's not a camera artifact]
        START_TIMESTAMP_MS: [start_timestamp_ms]
        END_TIMESTAMP_MS: [end_timestamp_ms]
        BEST_VISIBILITY_TIMESTAMP_MS: [best_visibility_timestamp_ms]
        LANDMARK_OBSERVATION_END

        If no clear candidates are found in the entire video/segment (that are not camera artifacts), state "No potential Landmarks found in this segment."
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

    def analyze_video_segment(self, gemini_response_text: str ,segment_state: PreprocessedVideoSegmentState) -> AnalyzedVideoSegmentState:
        if not self.gemini_model:
            return AnalyzedVideoSegmentState(
                processed_segment_info=segment_state,
                gemini_full_video_analysis_text="Error: Gemini model not available",
                identified_landmark_observations=[]
            )

        print(f"AnalystAgent: Extracting data from {segment_state['video_segment_path']} for the mission {segment_state['mission_id']}...")
        
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

    def run(self, video_segments: List[PreprocessedVideoSegmentState]) -> List[AnalyzedVideoSegmentState]:
        print(f"Analyst Agent: Starting analysis for {len(video_segments)} segment(s)...")
        analyzed_segments_list: List[AnalyzedVideoSegmentState] = []
        segments_prompt_video = [(self._build_prompt_for_video_analysis(segment_state), open(segment_state["video_segment_path"], 'rb').read()) \
                                 for segment_state in video_segments]
        
        # running asynchronously
        gemini_responses = asyncio.run(self.analysis_responses(segments_prompt_video))

        for segment, response in zip(video_segments, gemini_responses):
            analyzed_segment = self.analyze_video_segment(
                gemini_response_text=response.text, 
                segment_state=segment
            )
            analyzed_segments_list.append(analyzed_segment)
        print(f"Analyst Agent: Analysis complete.")
        return analyzed_segments_list