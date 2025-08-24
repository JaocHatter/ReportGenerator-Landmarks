import google.genai as genai
from google.genai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GeminiService:
    """
    A dedicated service class to encapsulate all interactions with the Google Gemini API.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.model = genai.Client(api_key = api_key)
        self.model_name = model_name
        logger.info(f"GeminiService initialized with model: {model_name}")

    def _build_contextual_analysis_prompt(self) -> str:
        """
        Constructs the detailed prompt for landmark analysis, similar to the IdentifierAgent.
        """
        return """
        You are a highly precise analytical AI for a Mars rover mission.
        An image of a potential landmark is provided.
        Your task is to provide a SUCCINCT and TECHNICAL analysis.
        You MUST STRICTLY follow this format, without any markdown or extra text. Be brief.

        OBJECT_NAME: [Specific, technical name for the object. E.g., "Handheld geological drill," "Scientific module control panel," "Heat shield fragment."]
        DETAILED_DESCRIPTION: [A single, concise sentence describing the object's key physical features. Focus on material, shape, and condition.]
        CONTEXTUAL_ANALYSIS: [A brief, point-form analysis. Answer each point in a single, short sentence.]
        - Probable origin:
        - Potential utility:
        - Relevance/Importance:
        """

    def _parse_contextual_response(self, response_text: str) -> Dict[str, str]:
        """
        Parses the structured text response from the Gemini model.
        """
        parsed_data = {
            "object_name": "Unknown",
            "description": "No description available.",
            "analysis": "No analysis available."
        }
        current_section = None
        
        lines = response_text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("OBJECT_NAME:"):
                parsed_data["object_name"] = line_stripped.split(":", 1)[1].strip()
                current_section = "description"
            elif line_stripped.startswith("DETAILED_DESCRIPTION:"):
                parsed_data["description"] = line_stripped.split(":", 1)[1].strip()
                current_section = "analysis"
            elif line_stripped.startswith("CONTEXTUAL_ANALYSIS:"):
                # Join all subsequent lines for the analysis
                analysis_text = line_stripped.split(":", 1)[1].strip()
                if i + 1 < len(lines):
                    analysis_text += "\n" + "\n".join(l.strip() for l in lines[i+1:])
                parsed_data["analysis"] = analysis_text.strip()
                break # Stop after capturing the full analysis
        
        return parsed_data

    async def get_contextual_analysis(self, image_bytes: bytes) -> Optional[Dict[str, str]]:
        """
        Analyzes an image to identify and contextually describe a landmark.
        
        Args:
            image_bytes: The image data in bytes.
            
        Returns:
            A dictionary containing the parsed analysis or None on failure.
        """
        prompt = self._build_contextual_analysis_prompt()

        try:
            response = await self.model.aio.models.generate_content(
                model = self.model_name,
                contents = [
                    types.Part(
                        inline_data = types.Blob(
                            data = image_bytes,
                            mime_type = 'image/jpeg'
                        )
                    ),
                    types.Part(text = prompt)
                ],
                config = types.GenerateContentConfig(
                    temperature = 0.2
                )
            )
            logger.info("Received response from Gemini API.")
            return self._parse_contextual_response(response.text)
            
        except Exception as e:
            logger.error(f"An error occurred while calling the Gemini API: {e}")
            return None
