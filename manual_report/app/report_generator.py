import time
import os
from typing import List, Dict, Any, Tuple
import markdown2
from weasyprint import HTML, CSS
import base64
import mimetypes
import matplotlib.pyplot as plt

class ReportGenerator:
    """
    Class to handle the generation of mission reports, including map creation and PDF conversion.
    """
    def __init__(self, landmarks_data: List[Dict[str, Any]], trajectory_data_path: str):
        self.REPORTS_DIR = "output"
        self.ASSETS_DIR = "assets"
        self.LOGO_PATH = os.path.join(self.ASSETS_DIR, "logo.png")
        
        self.landmarks = landmarks_data
        self.trajectory_path = trajectory_data_path
        self.mission_id = f"MISSION_{time.strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(self.REPORTS_DIR, exist_ok=True)
        
        self.map_filepath = os.path.join(self.REPORTS_DIR, f"temp_map_{self.mission_id}.png")
        self.pdf_filepath = os.path.join(self.REPORTS_DIR, f"Report_{self.mission_id}.pdf")

    def _read_trajectory_data(self) -> Tuple[List[float], List[float]]:
        """Reads trajectory data from the specified file."""
        path_x, path_y = [], []
        if not os.path.exists(self.trajectory_path):
            print(f"Warning: Trajectory file not found at {self.trajectory_path}")
            return path_x, path_y
        try:
            with open(self.trajectory_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        path_x.append(float(parts[0]))
                        path_y.append(float(parts[1]))
        except Exception as e:
            print(f"Error reading trajectory file: {e}")
        return path_x, path_y

    def _image_to_base64_uri(self, filepath: str) -> str:
        """Converts an image file to a Base64 Data URI."""
        try:
            mime_type, _ = mimetypes.guess_type(filepath)
            if not mime_type: mime_type = "application/octet-stream"
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"Could not convert image to Base64: {e}")
            return ""

    def _generate_mission_map(self, path_x: List[float], path_y: List[float]):
        """Generates and saves a 2D plot of the mission."""
        fig, ax = plt.subplots(figsize=(12, 10))
        if path_x and path_y:
            ax.plot(path_x, path_y, color='gray', linestyle='--', linewidth=1.5, label='Rover Path')
        if self.landmarks:
            lm_x = [lm['location']['x'] for lm in self.landmarks]
            lm_y = [lm['location']['y'] for lm in self.landmarks]
            labels = [lm['id'] for lm in self.landmarks]
            ax.scatter(lm_x, lm_y, c='black', marker='X', s=100, label='Landmarks', zorder=5)
            for i, txt in enumerate(labels):
                ax.annotate(txt, (lm_x[i], lm_y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.set_title('Mission Operations Map', fontsize=16)
        ax.set_xlabel('X Coordinate (meters)')
        ax.set_ylabel('Y Coordinate (meters)')
        ax.grid(True, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        fig.tight_layout()
        plt.savefig(self.map_filepath, dpi=150)
        plt.close()
        print(f"Mission map generated: {self.map_filepath}")

    def _generate_markdown_report(self) -> str:
        """Generates the full mission report in Markdown format."""
        report_lines = [f"# ERC 2025 Mission Report: {self.mission_id}", f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", "<!-- TOC_PLACEHOLDER -->"]
        
        report_lines.append(f"\n## Mission Summary")
        report_lines.append(f"- **Total Confirmed Landmarks:** {len(self.landmarks)}")
        if os.path.exists(self.map_filepath):
            map_uri = self._image_to_base64_uri(self.map_filepath)
            report_lines.append(f'\n### Operations Map\n<img src="{map_uri}" alt="Mission Overview Map" class="map-image">')
        
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
        """Converts the Markdown content to a PDF file."""
        logo_uri = self._image_to_base64_uri(self.LOGO_PATH) if os.path.exists(self.LOGO_PATH) else ""
        css_style = f"""
            @page {{ size: A4; margin: 1in; @top-left {{ content: 'ERC 2025 Mission Report'; font-size: 9pt; color: #888; }} @top-right {{ content: url('{logo_uri}'); transform: scale(0.4); position: absolute; top: -20px; right: 0; }} @bottom-center {{ content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #888; }} }}
            body {{ font-family: 'Helvetica', sans-serif; font-size: 11pt; line-height: 1.4; }} 
            h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 25px; }} 
            h2 {{ page-break-before: always; border-bottom: 1px solid #ccc; padding-top: 15px; font-size: 16pt; bookmark-level: 1; }} 
            h3 {{ font-size: 12pt; font-weight: bold; margin-bottom: -5px; }}
            img {{ display: block; margin: 20px auto; max-width: 60%; max-height: 35vh; border: 1px solid #ddd; padding: 4px; }} 
            .map-image {{ max-width: 95%; max-height: none; }}
            blockquote {{ margin-left: 15px; padding-left: 15px; border-left: 3px solid #eee; font-style: italic; color: #333; }}
            .toc-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 30px; page-break-after: always; }}
            .toc-table h2 {{ page-break-before: auto !important; border-bottom: 1px solid #ccc; padding-bottom: 5px; font-size: 16pt; margin-bottom: 15px; }}
            .toc-table td {{ padding: 6px 0; border-bottom: 1px dotted #999; }}
            .toc-table td:last-child {{ text-align: right; font-weight: bold; }}
        """
        
        # Pass 1: Render to find page numbers
        html_body_pass1 = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'markdown-in-html', 'header-ids'])
        full_html_pass1 = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_body_pass1}</body></html>"
        doc = HTML(string=full_html_pass1).render(stylesheets=[CSS(string=css_style)])
        
        toc_entries = []
        if hasattr(doc, 'bookmarks'):
            for level, label, (page_num, dest_x, dest_y) in doc.bookmarks:
                if label.startswith("Landmark:"):
                    landmark_id = label.replace("Landmark:", "").strip()
                    toc_entries.append({'id': landmark_id, 'page': page_num})

        # Pass 2: Build the TOC and render the final PDF
        toc_html = ""
        if toc_entries:
            toc_html = '<div class="toc-table"><h2>Table of Contents</h2><table>'
            for entry in toc_entries:
                toc_html += f"<tr><td>{entry['id']}</td><td>{entry['page']}</td></tr>"
            toc_html += '</table></div>'

        final_html_body = html_body_pass1.replace("<!-- TOC_PLACEHOLDER -->", toc_html)
        full_final_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{final_html_body}</body></html>"
        
        HTML(string=full_final_html).write_pdf(self.pdf_filepath, stylesheets=[CSS(string=css_style)])
        print(f"✅ PDF report generated successfully: {self.pdf_filepath}")

    def generate_report(self) -> str:
        """Executes the full report generation pipeline."""
        path_x, path_y = self._read_trajectory_data()
        self._generate_mission_map(path_x, path_y)
        md_content = self._generate_markdown_report()
        self._convert_md_to_pdf(md_content)
        if os.path.exists(self.map_filepath):
            os.remove(self.map_filepath)
            print(f"Temporary map file removed: {self.map_filepath}")
        return self.pdf_filepath
