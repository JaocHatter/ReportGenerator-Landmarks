import os
from typing import List, Optional
from states import (
    IdentifiedLandmarksBatchState, ConfirmedLandmarkState, RobotPose
)

import matplotlib
matplotlib.use('Agg') # To prevent GUI issues on servers
import matplotlib.pyplot as plt

class ReportGeneratorAgent:
    def __init__(self, output_dir: str = "output/reports", map_image_dir: str = "output/map_images", landmark_image_dir: str = "output/landmark_images"):
        self.output_dir = output_dir
        self.map_image_dir = map_image_dir
        self.landmark_image_dir = landmark_image_dir 

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.map_image_dir, exist_ok=True)

    def _generate_map_image(self, robot_path: List[RobotPose], landmarks: List[ConfirmedLandmarkState], mission_id: str) -> Optional[str]:
        """Generates a simple map image and saves it, returning the relative path for Markdown."""
        map_filename = f"map_{mission_id}.png"
        map_abs_path = os.path.join(self.map_image_dir, map_filename)
        map_relative_path = os.path.join("..", "map_images", map_filename)


        plt.figure(figsize=(10, 8)) # Slightly larger for better detail
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
                plt.annotate(txt, (lm_x[i], lm_y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='black')

        plt.title(f"Marsyard Map - Mission: {mission_id}", fontsize=16)
        plt.xlabel("X Coordinate (m)", fontsize=12)
        plt.ylabel("Y Coordinate (m)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout() # Adjust layout so everything fits well
        try:
            plt.savefig(map_abs_path)
            plt.close()
            print(f"Map generated and saved to: {map_abs_path}")
            return map_relative_path
        except Exception as e:
            print(f"Error generating or saving map: {e}")
            plt.close()
            return None # Indicate that the map could not be generated

    def _prepare_markdown_content(self, batch_state: IdentifiedLandmarksBatchState) -> List[str]:
        """Prepares the report content as a list of Markdown strings."""
        markdown_lines: List[str] = []
        mission_id = batch_state['mission_id']

        markdown_lines.append(f"# ERC 2025 Mission Report: {mission_id}")
        markdown_lines.append("\n## General Findings\n")
        markdown_lines.append(f"- **Total Landmarks Found:** {len(batch_state['confirmed_landmarks'])}")
        
        llm_summary = "Mission summary by LLM (implementation with Gemini pending)."
        markdown_lines.append(f"- **Mission Summary (LLM):** {llm_summary}\n")

        markdown_lines.append("### Mission Map\n")
        map_relative_path = self._generate_map_image(
            batch_state['full_robot_path_poses'],
            batch_state['confirmed_landmarks'],
            mission_id
        )
        if map_relative_path:
            map_display_path = map_relative_path.replace("\\", "/")
            markdown_lines.append(f"![Mission Map]({map_display_path})\n")
        else:
            markdown_lines.append("*Could not generate map image.*\n")

        markdown_lines.append("\n---\n---\n") # More prominent separator

        # --- Individual Landmark Sections ---
        if not batch_state['confirmed_landmarks']:
            markdown_lines.append("\n**No landmarks confirmed in this mission.**\n")
        
        for lm in batch_state['confirmed_landmarks']:
            markdown_lines.append(f"\n## Landmark Detail: {lm['landmark_id']}\n")

            # Landmark image path
            # We assume lm['best_image_path'] is an absolute path or relative to the execution directory.
            # We need to make it relative to the Markdown report directory.
            # If lm['best_image_path'] is 'output/landmark_images/LM_mission_001.jpg'
            # and the report is in 'output/reports/report.md', the relative path is '../landmark_images/...'
            if lm['best_image_path'] and os.path.exists(lm['best_image_path']):
                try:
                    # Build relative path from output_dir to landmark_image_dir
                    # landmark_image_relative_path = os.path.relpath(lm['best_image_path'], self.output_dir)
                    # This is simpler if we assume a fixed structure:
                    landmark_image_filename = os.path.basename(lm['best_image_path'])
                    landmark_image_display_path = os.path.join("..", "landmark_images", landmark_image_filename).replace("\\", "/")
                    markdown_lines.append(f"![Photo of Landmark {lm['landmark_id']}]({landmark_image_display_path})\n")
                except Exception as e_img:
                     markdown_lines.append(f"*Could not generate link for landmark photo {lm['landmark_id']}: {e_img} (Original path: {lm['best_image_path']})*\n")
            else:
                markdown_lines.append(f"*Photo of landmark {lm['landmark_id']} not available or incorrect path ({lm['best_image_path']}).*\n")

            markdown_lines.append(f"- **Name/Category:** {lm['object_name_or_category']}")
            
            # Handle multiline descriptions and analyses
            markdown_lines.append("- **Detailed Visual Description:**")
            for line in lm['detailed_visual_description'].split('\n'):
                markdown_lines.append(f"  {line}") # Indented for clarity as sub-item

            markdown_lines.append("- **Martian Contextual Analysis:**")
            for line in lm['contextual_analysis'].split('\n'):
                markdown_lines.append(f"  {line}")

            markdown_lines.append("- **Estimated Location (Robot Pose):**")
            markdown_lines.append(f"    - Timestamp: {lm['estimated_location']['timestamp_ms']} ms")
            markdown_lines.append(f"    - X: {lm['estimated_location']['x']:.2f} m, Y: {lm['estimated_location']['y']:.2f} m")
            markdown_lines.append(f"    - Orientation: {lm['estimated_location']['orientation_degrees']:.1f}°")
            
            markdown_lines.append("\n---\n") # Separator between landmarks

        return markdown_lines

    def generate_markdown_report(self, markdown_content: List[str], mission_id: str) -> str:
        """Generates the Markdown report and saves it to a .md file."""
        report_filename_md = f"ERC2025_Report_{mission_id}.md"
        report_filepath = os.path.join(self.output_dir, report_filename_md)

        try:
            with open(report_filepath, "w", encoding="utf-8") as f:
                for line in markdown_content:
                    f.write(line + "\n") # Ensure new line for each list item
            print(f"Markdown report generated: {report_filepath}")
            return report_filepath
        except Exception as e:
            print(f"Error writing Markdown file: {e}")
            return "" # Return empty string or None in case of error

    def run(self, identified_landmarks_batch: IdentifiedLandmarksBatchState) -> str:
        """
        Orchestrates content preparation and Markdown file generation.
        Returns the path to the generated Markdown file.
        """
        if not identified_landmarks_batch or not identified_landmarks_batch.get('mission_id'):
            print("Report Generator Agent: Invalid input data or missing mission_id.")
            return "Could not generate report due to invalid input data."

        print(f"Report Generator Agent (Markdown): Starting for mission {identified_landmarks_batch['mission_id']}...")
        
        markdown_content_list = self._prepare_markdown_content(identified_landmarks_batch)
        report_file_path = self.generate_markdown_report(markdown_content_list, identified_landmarks_batch['mission_id'])
        
        print(f"Report Generator Agent (Markdown): Finished.")
        return report_file_path