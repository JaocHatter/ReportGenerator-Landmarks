import os
import random
from typing import List, Optional, Dict, TypedDict

# --- Library Imports ---
import markdown2
from weasyprint import HTML, CSS
import matplotlib
matplotlib.use('Agg')  # To prevent GUI issues on servers
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

#
# -------------------- SECTION 1: DATA STRUCTURES (TypedDicts) --------------------
#
class RobotPose(TypedDict):
    """Represents the robot's pose at a specific time."""
    timestamp_ms: int
    x: float
    y: float
    orientation_degrees: float

class ConfirmedLandmarkState(TypedDict):
    """Represents a fully analyzed and confirmed landmark."""
    landmark_id: str
    mission_id: str
    best_image_path: str  # Path to the best saved image of the landmark
    object_name_or_category: str
    detailed_visual_description: str
    contextual_analysis: str
    estimated_location: RobotPose
    frames_observed_timestamps: List[int]

class IdentifiedLandmarksBatchState(TypedDict):
    """Contains all data for a mission to be included in the final report."""
    mission_id: str
    confirmed_landmarks: List[ConfirmedLandmarkState]
    full_robot_path_poses: List[RobotPose]


#
# -------------------- SECTION 2: THE REPORT GENERATOR AGENT CLASS --------------------
#
class ReportGeneratorAgent:
    def __init__(self, 
                 output_dir: str = "output/reports", 
                 map_image_dir: str = "output/map_images", 
                 landmark_image_dir: str = "output/landmark_images",
                 generate_pdf: bool = True):
        self.output_dir = output_dir
        self.map_image_dir = map_image_dir
        self.landmark_image_dir = landmark_image_dir
        self.generate_pdf = generate_pdf

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.map_image_dir, exist_ok=True)
        os.makedirs(self.landmark_image_dir, exist_ok=True) # Ensure landmark image dir exists

    def _convert_md_to_pdf(self, markdown_filepath: str, pdf_filepath: str):
        """Converts a given Markdown file to a PDF with sections on new pages."""
        print(f"Starting PDF conversion for: {markdown_filepath}")
        
        try:
            with open(markdown_filepath, 'r', encoding='utf-8') as f:
                md_content = f.read()

            css_style = """
            @page { size: A4; margin: 1in; }
            body { font-family: 'Helvetica', sans-serif; line-height: 1.6; font-size: 11pt; }
            h1 { text-align: center; border-bottom: 3px solid #004a80; padding-bottom: 15px; color: #004a80; font-size: 24pt; }
            h2 { page-break-before: always; border-bottom: 1.5px solid #cccccc; padding-top: 15px; color: #004a80; font-size: 18pt; }
            h3 { color: #333333; font-size: 14pt; margin-top: 25px; }
            img { max-width: 90%; height: auto; display: block; margin-left: auto; margin-right: auto; border: 1px solid #ddd; padding: 5px; border-radius: 4px; }
            code { background-color: #f0f0f0; padding: 2px 5px; border-radius: 4px; font-family: 'Courier New', monospace; }
            ul { list-style-type: disc; padding-left: 20px; }
            """
            
            html_body = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables', 'cuddled-lists'])
            full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{css_style}</style></head><body>{html_body}</body></html>"

            base_url = os.path.dirname(os.path.abspath(markdown_filepath))
            HTML(string=full_html, base_url=base_url).write_pdf(pdf_filepath)
            
            print(f"✅ PDF report generated successfully: {pdf_filepath}")

        except Exception as e:
            print(f"❌ Error generating PDF: {e}")

    def _generate_map_image(self, robot_path: List[RobotPose], landmarks: List[ConfirmedLandmarkState], mission_id: str) -> Optional[str]:
        map_filename = f"map_{mission_id}.png"
        map_abs_path = os.path.join(self.map_image_dir, map_filename)
        map_relative_path = os.path.join("..", self.map_image_dir.split('/')[-1], map_filename)

        plt.figure(figsize=(10, 8))
        if robot_path:
            path_x = [p['x'] for p in robot_path]
            path_y = [p['y'] for p in robot_path]
            plt.plot(path_x, path_y, marker='.', linestyle='-', label="Robot Path", markersize=3, linewidth=1, color='cornflowerblue')

        if landmarks:
            lm_x = [lm['estimated_location']['x'] for lm in landmarks]
            lm_y = [lm['estimated_location']['y'] for lm in landmarks]
            lm_ids = [lm['landmark_id'] for lm in landmarks]
            plt.scatter(lm_x, lm_y, c='red', marker='X', s=120, label="Landmarks", edgecolors='black', linewidth=0.5)
            for i, txt in enumerate(lm_ids):
                plt.annotate(txt, (lm_x[i], lm_y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

        plt.title(f"Marsyard Map - Mission: {mission_id}", fontsize=16)
        plt.xlabel("X Coordinate (m)", fontsize=12)
        plt.ylabel("Y Coordinate (m)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        try:
            plt.savefig(map_abs_path)
            plt.close()
            return map_relative_path
        except Exception as e:
            print(f"Error generating or saving map: {e}")
            plt.close()
            return None

    def _prepare_markdown_content(self, batch_state: IdentifiedLandmarksBatchState) -> str:
        """Prepares the full report content as a single Markdown string."""
        markdown_lines: List[str] = []
        mission_id = batch_state['mission_id']

        markdown_lines.append(f"# ERC 2025 Mission Report: {mission_id}")
        markdown_lines.append(f"\n*Report generated on: {matplotlib.dates.datetime.date.today().strftime('%Y-%m-%d')}*")
        
        markdown_lines.append("\n## General Findings")
        markdown_lines.append(f"- **Total Landmarks Found:** {len(batch_state['confirmed_landmarks'])}")
        llm_summary = "The mission was successful, identifying key geological and artificial markers. The robot navigated the terrain efficiently, covering all designated waypoints."
        markdown_lines.append(f"- **Mission Summary (LLM):** {llm_summary}\n")

        markdown_lines.append("### Mission Map\n")
        map_relative_path = self._generate_map_image(
            batch_state['full_robot_path_poses'], batch_state['confirmed_landmarks'], mission_id
        )
        if map_relative_path:
            map_display_path = map_relative_path.replace("\\", "/")
            markdown_lines.append(f"![Mission Map]({map_display_path})\n")
        else:
            markdown_lines.append("*Could not generate map image.*\n")

        if not batch_state['confirmed_landmarks']:
            markdown_lines.append("\n**No landmarks confirmed in this mission.**\n")
        
        for lm in batch_state['confirmed_landmarks']:
            markdown_lines.append(f"\n## Landmark Detail: {lm['landmark_id']}")

            if lm['best_image_path'] and os.path.exists(lm['best_image_path']):
                lm_img_filename = os.path.basename(lm['best_image_path'])
                lm_img_display_path = os.path.join("..", self.landmark_image_dir.split('/')[-1], lm_img_filename).replace("\\", "/")
                markdown_lines.append(f"![Photo of Landmark {lm['landmark_id']}]({lm_img_display_path})\n")
            else:
                markdown_lines.append(f"*Photo not available for landmark {lm['landmark_id']} (Path: {lm['best_image_path']}).*\n")

            markdown_lines.append(f"- **Name/Category:** {lm['object_name_or_category']}")
            markdown_lines.append("- **Detailed Visual Description:**")
            for line in lm['detailed_visual_description'].split('\n'):
                if line.strip(): markdown_lines.append(f"  > {line}")
            
            markdown_lines.append("- **Martian Contextual Analysis:**")
            for line in lm['contextual_analysis'].split('\n'):
                if line.strip(): markdown_lines.append(f"  > {line}")

            markdown_lines.append("- **Estimated Location (Robot Pose):**")
            markdown_lines.append(f"  - Timestamp: {lm['estimated_location']['timestamp_ms']} ms")
            markdown_lines.append(f"  - X: {lm['estimated_location']['x']:.2f} m, Y: {lm['estimated_location']['y']:.2f} m")
            markdown_lines.append(f"  - Orientation: {lm['estimated_location']['orientation_degrees']:.1f}°")

        return "\n".join(markdown_lines)

    def generate_markdown_report(self, markdown_content: str, mission_id: str) -> str:
        """Saves the Markdown content to a .md file and returns the path."""
        report_filename_md = f"ERC2025_Report_{mission_id}.md"
        report_filepath = os.path.join(self.output_dir, report_filename_md)
        try:
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"📄 Markdown report generated: {report_filepath}")
            return report_filepath
        except Exception as e:
            print(f"❌ Error writing Markdown file: {e}")
            return ""

    def run(self, identified_landmarks_batch: IdentifiedLandmarksBatchState) -> Dict[str, Optional[str]]:
        if not identified_landmarks_batch or not identified_landmarks_batch.get('mission_id'):
            print("Report Generator Agent: Invalid input data or missing mission_id.")
            return {"markdown": None, "pdf": None}
        
        mission_id = identified_landmarks_batch['mission_id']
        print(f"--- Report Generator Agent: Starting for mission {mission_id} ---")
        
        markdown_full_content = self._prepare_markdown_content(identified_landmarks_batch)
        md_report_path = self.generate_markdown_report(markdown_full_content, mission_id)
        
        pdf_report_path = None
        if self.generate_pdf and md_report_path:
            pdf_report_path = md_report_path.replace(".md", ".pdf")
            self._convert_md_to_pdf(md_report_path, pdf_report_path)
        
        print(f"--- Report Generator Agent: Finished mission {mission_id}. ---")
        return {"markdown": md_report_path, "pdf": pdf_report_path}

#
# -------------------- SECTION 3: TEST DATA AND DUMMY IMAGE GENERATION --------------------
#
def create_dummy_image(path: str, text: str):
    """Creates a simple placeholder image with text."""
    if os.path.exists(path):
        return
    
    img = Image.new('RGB', (400, 300), color=(70, 130, 180)) # SteelBlue
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default() # Fallback font
    
    d.text((10,10), text, fill=(255,255,255), font=font)
    img.save(path)
    print(f"🖼️  Created dummy image: {path}")

def generate_test_data(mission_id: str) -> IdentifiedLandmarksBatchState:
    """Generates a populated IdentifiedLandmarksBatchState for testing."""
    print("Generating test data...")
    
    # 1. Create a robot path
    path = []
    for i in range(20):
        path.append({
            "timestamp_ms": 1000 * i, "x": i * 0.5, "y": 2 * (i % 5), "orientation_degrees": float(i * 10 % 360)
        })

    # 2. Define confirmed landmarks
    landmarks = [
        {
            "landmark_id": "LM-01",
            "mission_id": mission_id,
            "best_image_path": f"output/landmark_images/{mission_id}_LM-01.jpg",
            "object_name_or_category": "Damaged Solar Panel",
            "detailed_visual_description": "A rectangular solar panel, approximately 1.5m x 1m. The top glass is cracked extensively across the upper-left quadrant.",
            "contextual_analysis": "This appears to be part of a previous uncrewed mission. The damage could be from micrometeorite impact. It is not a natural formation.",
            "estimated_location": path[5],
            "frames_observed_timestamps": [4800, 5000, 5200]
        },
        {
            "landmark_id": "LM-02",
            "mission_id": mission_id,
            "best_image_path": f"output/landmark_images/{mission_id}_LM-02.jpg",
            "object_name_or_category": "Unusual Rock Formation",
            "detailed_visual_description": "A dark, igneous-type rock with visible crystalline structures. It has an unusual, almost perfectly spherical shape, about 30cm in diameter.",
            "contextual_analysis": "The spherical nature is highly unusual for the surrounding geology. It could be a meteorite or a concretion formed by water action in the distant past. Recommend further analysis.",
            "estimated_location": path[15],
            "frames_observed_timestamps": [14800, 15000, 15200]
        }
    ]

    # 3. Assemble the final state object
    batch_state: IdentifiedLandmarksBatchState = {
        "mission_id": mission_id,
        "confirmed_landmarks": landmarks,
        "full_robot_path_poses": path
    }
    print("Test data generated.")
    return batch_state


#
# -------------------- SECTION 4: MAIN EXECUTION BLOCK --------------------
#
if __name__ == "__main__":
    TEST_MISSION_ID = "mission_2025_alpha"
    
    # 1. Generate the structured data for the report
    test_batch_state = generate_test_data(TEST_MISSION_ID)

    # 2. Create the dummy landmark image files so the report can find them
    #    This simulates the vision system saving landmark images.
    os.makedirs("output/landmark_images", exist_ok=True)
    for landmark in test_batch_state["confirmed_landmarks"]:
        create_dummy_image(landmark["best_image_path"], f"This is a photo of:\n{landmark['landmark_id']}")
    
    # 3. Instantiate the agent and run the full process
    report_agent = ReportGeneratorAgent(
        output_dir="output/reports",
        map_image_dir="output/map_images",
        landmark_image_dir="output/landmark_images"
    )
    
    # 4. Execute the run method
    generated_files = report_agent.run(test_batch_state)

    # 5. Print the final result
    print("\n--- 🚀 Process Complete ---")
    if generated_files.get("markdown"):
        print(f"Markdown report is at: {generated_files['markdown']}")
    if generated_files.get("pdf"):
        print(f"PDF report is at:      {generated_files['pdf']}")
    print("--------------------------")