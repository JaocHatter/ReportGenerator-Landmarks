import requests
import time
import os
from typing import List, Dict, Any
import markdown2
from weasyprint import HTML, CSS
import base64
import mimetypes

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
LANDMARKS_ENDPOINT = f"{API_BASE_URL}/landmarks/"
REPORTS_DIR = "mission_reports"

def fetch_landmarks_from_api() -> List[Dict[str, Any]]:
    """
    Fetches the complete list of landmarks from our running API.
    """
    print(f"Attempting to fetch data from {LANDMARKS_ENDPOINT}...")
    try:
        response = requests.get(LANDMARKS_ENDPOINT)
        response.raise_for_status()
        print("Successfully fetched data.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR ---")
        print(f"Could not connect to the Mars Landmark API.")
        print(f"Please ensure the FastAPI server is running.")
        print(f"Details: {e}")
        print("---------------")
        return []

def convert_md_to_pdf(markdown_filepath: str, pdf_filepath: str):
    """
    Converts a given Markdown file to a PDF with specific styling.
    """
    print(f"Starting PDF conversion for: {markdown_filepath}")
    try:
        with open(markdown_filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()

        css_style = """
        @page {
            size: A4;
            margin: 1in;
        }
        body {
            font-family: 'Helvetica', sans-serif;
            line-height: 1.6;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        h2 {
            page-break-before: always;
            border-bottom: 1px solid #ccc;
            padding-top: 15px;
        }
        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 60%;
            max-height: 45vh;
            height: auto;
            border: 1px solid #ddd;
            padding: 4px;
        }
        """
        
        html_body = markdown2.markdown(md_content, extras=['fenced-code-blocks'])
        full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_body}</body></html>"
        
        HTML(string=full_html).write_pdf(pdf_filepath, stylesheets=[CSS(string=css_style)])
        
        print(f"✅ PDF report generated successfully: {pdf_filepath}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

def image_to_base64_uri(filepath: str) -> str:
    """Reads an image file and converts it to a Base64 Data URI."""
    try:
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type:
            mime_type = "application/octet-stream" # Fallback
            
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Could not convert image to Base64: {e}")
        return ""

def generate_markdown_report(landmarks: List[Dict[str, Any]]) -> str:
    """
    Generates a full mission report in Markdown format by embedding images as Base64.
    """
    if not landmarks:
        return "# Mission Report\n\nNo landmarks were processed."

    mission_id = f"MISSION_{time.strftime('%Y%m%d')}"
    report_lines = [f"# ERC 2025 Mission Report: {mission_id}"]
    report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report_lines.append(f"\n## Summary\n- **Total Landmarks Confirmed:** {len(landmarks)}")

    for lm in landmarks:
        report_lines.append(f"\n## Landmark Detail: {lm.get('id', 'N/A')}")
        
        image_path = lm.get('best_image_path')
        if image_path and os.path.exists(image_path):
            # --- FIX: Convert image to Base64 and embed it directly ---
            base64_uri = image_to_base64_uri(image_path)
            if base64_uri:
                report_lines.append(f"\n![Photo of {lm.get('name', 'N/A')}]({base64_uri})\n")
            else:
                report_lines.append(f"\n*Failed to embed image from path: {image_path}*\n")
        else:
            report_lines.append(f"\n*Image not available at path: {image_path}*\n")
        
        report_lines.append(f"### Name/Category\n**{lm.get('name', 'N/A')}**")
        
        ts = lm.get('timestamp', 0)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(ts))
        report_lines.append(f"### Observation Timestamp\n{formatted_time}")
        
        loc = lm.get('location', {})
        pos_str = f"X={loc.get('x', 0):.2f}m, Y={loc.get('y', 0):.2f}m, Z={loc.get('z', 0):.2f}m"
        report_lines.append(f"### Estimated Location\n{pos_str}")
        
        desc = lm.get('detailed_description', 'Not provided.').replace('\n', '\n> ')
        report_lines.append(f"### Detailed Visual Description\n> {desc}")
        
        analysis = lm.get('contextual_analysis', 'Not provided.').replace('\n', '\n> ')
        report_lines.append(f"### Martian Contextual Analysis\n> {analysis}")
        
    return "\n".join(report_lines)

def save_report_to_file(report_content: str, mission_id: str) -> str:
    """Saves the generated report content to a Markdown file."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    filename = f"ERC2025_Report_{mission_id}_{int(time.time())}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n✅ Markdown report successfully generated: {filepath}")
        return filepath
    except IOError as e:
        print(f"\nError: Could not save report file. Details: {e}")
        return ""

def main():
    """
    Main function to fetch data, generate MD and PDF reports.
    """
    landmarks = fetch_landmarks_from_api()
    if landmarks:
        mission_id = f"MISSION_{time.strftime('%Y%m%d')}"
        report_content = generate_markdown_report(landmarks)
        md_filepath = save_report_to_file(report_content, mission_id)
        
        if md_filepath:
            pdf_filepath = md_filepath.replace(".md", ".pdf")
            convert_md_to_pdf(md_filepath, pdf_filepath)
    else:
        print("No landmarks found. Report not generated.")

if __name__ == "__main__":
    main()
