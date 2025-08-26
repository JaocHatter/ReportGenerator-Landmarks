import time
import os
from typing import List, Dict, Any
import markdown2
from weasyprint import HTML, CSS
import base64
import mimetypes
import json
from app.map_marker import MapAnnotator

class ReportGenerator:
    def __init__(self, landmarks_data: List[Dict[str, Any]], map_files: Dict[str, str]):
        self.REPORTS_DIR = "output"
        self.ASSETS_DIR = "assets"
        self.LOGO_PATH = os.path.join(self.ASSETS_DIR, "logo.png")
        
        self.landmarks = landmarks_data
        self.map_files = map_files
        self.mission_id = f"MISSION_{time.strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(self.REPORTS_DIR, exist_ok=True)
        
        self.map_filepath = os.path.join(self.REPORTS_DIR, f"AnnotatedMap_{self.mission_id}.png")
        self.pdf_filepath = os.path.join(self.REPORTS_DIR, f"Report_{self.mission_id}.pdf")

    def _generate_annotated_map(self):
        temp_markers_path = None
        try:
            simple_landmarks = [
                {"name": lm.get('id', 'N/A'), "x": lm.get('location', {}).get('x'), "y": lm.get('location', {}).get('y')}
                for lm in self.landmarks
            ]
            temp_markers_path = os.path.join(self.REPORTS_DIR, f"temp_markers_{self.mission_id}.json")
            with open(temp_markers_path, 'w') as f:
                json.dump(simple_landmarks, f)

            annotator = MapAnnotator(
                yaml_path=self.map_files['yaml'],
                pgm_path=os.path.basename(self.map_files['pgm']) 
            )
            annotator.draw_trajectory(self.map_files['trajectory'])
            annotator.draw_markers(temp_markers_path)
            annotator.save_annotated_map(self.map_filepath)
            
            if not os.path.exists(self.map_filepath):
                raise FileNotFoundError("Annotated map could not be created.")
            
            print(f"Annotated mission map saved at: {self.map_filepath}")

        except Exception as e:
            print(f"Error generating annotated map: {e}")
            self.map_filepath = None
        finally:
            if temp_markers_path and os.path.exists(temp_markers_path):
                os.remove(temp_markers_path)

    def _image_to_base64_uri(self, filepath: str) -> str:
        try:
            abs_path = os.path.abspath(filepath)
            mime_type, _ = mimetypes.guess_type(abs_path)
            if not mime_type: mime_type = "application/octet-stream"
            with open(abs_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"Could not convert image to Base64: {e}")
            return ""

    def _generate_markdown_report(self) -> str:
        report_lines = [f"# ERC 2025 Mission Report: {self.mission_id}", f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}"]
        
        report_lines.append(f"\n## Mission Summary")
        report_lines.append(f"- **Total Confirmed Landmarks:** {len(self.landmarks)}")
        
        if self.map_filepath and os.path.exists(self.map_filepath):
            map_uri = self._image_to_base64_uri(self.map_filepath)
            report_lines.append(f'\n### Operations Map\n<img src="{map_uri}" alt="Mission Overview Map" class="map-image">')
        else:
            report_lines.append(f'\n### Operations Map\n*Map could not be generated.*')

        for lm in self.landmarks:
            lm_id = lm.get('id', 'N/A')
            report_lines.append(f"\n## Landmark: {lm_id}")
            image_path = lm.get('best_image_path')
            if image_path and os.path.exists(image_path):
                report_lines.append(f"\n![Photo of {lm.get('name', 'N/A')}]({self._image_to_base64_uri(image_path)})\n")
            else:
                report_lines.append(f"\n*Image not available.*\n")
            report_lines.extend([
                f"### Name/Category\n**{lm.get('name', 'N/A')}**",
                f"### Observation Timestamp\n{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(lm.get('timestamp', 0)))}",
                f"### Estimated Location\n`X={lm.get('location', {}).get('x', 0):.2f}m, Y={lm.get('location', {}).get('y', 0):.2f}m, Z={lm.get('location', {}).get('z', 0):.2f}m`",
                f"### Detailed Visual Description\n> {lm.get('detailed_description', 'Not provided.').replace(chr(10), chr(10) + '> ')}",
                f"### Martian Contextual Analysis\n> {lm.get('contextual_analysis', 'Not provided.').replace(chr(10), chr(10) + '> ')}"
            ])
        return "\n".join(report_lines)

    def _convert_md_to_pdf(self, md_content: str):
        logo_uri = self._image_to_base64_uri(self.LOGO_PATH) if os.path.exists(self.LOGO_PATH) else ""
        css_style = f"""
            @page {{ size: letter; margin: 1in; @top-left {{ content: 'ERC 2025 Mission Report'; font-size: 9pt; color: #888; }} @top-right {{ content: url('{logo_uri}'); transform: scale(0.4); position: absolute; top: -20px; right: 0; }} @bottom-center {{ content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #888; }} }}
            body {{ font-family: 'Helvetica', sans-serif; font-size: 10pt; line-height: 1.3; }} 
            h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 25px; }} 
            h2 {{ page-break-before: always; border-bottom: 1px solid #ccc; padding-top: 15px; font-size: 14pt; }} 
            h3 {{ font-size: 11pt; font-weight: bold; margin-bottom: -5px; }}
            img {{ display: block; margin: 10px auto; max-width: 60%; border: 1px solid #ddd; padding: 4px; }} 
            .map-image {{ max-width: 100%; page-break-inside: avoid; }}
            blockquote {{ margin-left: 15px; padding-left: 15px; border-left: 3px solid #eee; font-style: italic; color: #333; }}
        """
        html_body = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'markdown-in-html'])
        full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_body}</body></html>"
        HTML(string=full_html).write_pdf(self.pdf_filepath, stylesheets=[CSS(string=css_style)])
        print(f"✅ PDF report generated successfully: {self.pdf_filepath}")

    def generate_report(self) -> str:
        self._generate_annotated_map()
        md_content = self._generate_markdown_report()
        self._convert_md_to_pdf(md_content)
        return self.pdf_filepath
