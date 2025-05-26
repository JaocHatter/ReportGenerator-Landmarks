import os
from typing import List, Optional
from states import (
    IdentifiedLandmarksBatchState, ConfirmedLandmarkState, RobotPose
)

import matplotlib
matplotlib.use('Agg') # Para evitar problemas de GUI en servidores
import matplotlib.pyplot as plt

class ReportGeneratorAgent:
    def __init__(self, output_dir: str = "output/reports", map_image_dir: str = "output/map_images", landmark_image_dir: str = "output/landmark_images"):
        self.output_dir = output_dir
        self.map_image_dir = map_image_dir
        self.landmark_image_dir = landmark_image_dir 

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.map_image_dir, exist_ok=True)

    def _generate_map_image(self, robot_path: List[RobotPose], landmarks: List[ConfirmedLandmarkState], mission_id: str) -> Optional[str]:
        """Genera una imagen de mapa simple y la guarda, retornando el path relativo para Markdown."""
        map_filename = f"map_{mission_id}.png"
        map_abs_path = os.path.join(self.map_image_dir, map_filename)
        map_relative_path = os.path.join("..", "map_images", map_filename)


        plt.figure(figsize=(10, 8)) # Un poco más grande para mejor detalle
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
        plt.tight_layout() # Ajustar layout para que todo quepa bien
        try:
            plt.savefig(map_abs_path)
            plt.close()
            print(f"Mapa generado y guardado en: {map_abs_path}")
            return map_relative_path
        except Exception as e:
            print(f"Error al generar o guardar el mapa: {e}")
            plt.close()
            return None # Indicar que no se pudo generar el mapa

    def _prepare_markdown_content(self, batch_state: IdentifiedLandmarksBatchState) -> List[str]:
        """Prepara el contenido del reporte como una lista de strings Markdown."""
        markdown_lines: List[str] = []
        mission_id = batch_state['mission_id']

        markdown_lines.append(f"# Reporte de Misión ERC 2025: {mission_id}")
        markdown_lines.append("\n## Hallazgos Generales\n")
        markdown_lines.append(f"- **Total de Landmarks Encontrados:** {len(batch_state['confirmed_landmarks'])}")
        
        llm_summary = "Resumen de la misión por LLM (pendiente implementación con Gemini)."
        markdown_lines.append(f"- **Resumen de la Misión (LLM):** {llm_summary}\n")

        markdown_lines.append("### Mapa de la Misión\n")
        map_relative_path = self._generate_map_image(
            batch_state['full_robot_path_poses'],
            batch_state['confirmed_landmarks'],
            mission_id
        )
        if map_relative_path:
            map_display_path = map_relative_path.replace("\\", "/")
            markdown_lines.append(f"![Mapa de la Misión]({map_display_path})\n")
        else:
            markdown_lines.append("*No se pudo generar la imagen del mapa.*\n")

        markdown_lines.append("\n---\n---\n") # Separador más prominente

        # --- Secciones de Landmarks Individuales ---
        if not batch_state['confirmed_landmarks']:
            markdown_lines.append("\n**No se confirmaron landmarks en esta misión.**\n")
        
        for lm in batch_state['confirmed_landmarks']:
            markdown_lines.append(f"\n## Detalle del Landmark: {lm['landmark_id']}\n")

            # Path de la imagen del landmark
            # Asumimos que lm['best_image_path'] es un path absoluto o relativo al directorio de ejecución.
            # Necesitamos hacerlo relativo al directorio del reporte Markdown.
            # Si lm['best_image_path'] es 'output/landmark_images/LM_mission_001.jpg'
            # y el reporte está en 'output/reports/report.md', el path relativo es '../landmark_images/...'
            if lm['best_image_path'] and os.path.exists(lm['best_image_path']):
                try:
                    # Construir path relativo desde output_dir al landmark_image_dir
                    # landmark_image_relative_path = os.path.relpath(lm['best_image_path'], self.output_dir)
                    # Esto es más simple si asumimos una estructura fija:
                    landmark_image_filename = os.path.basename(lm['best_image_path'])
                    landmark_image_display_path = os.path.join("..", "landmark_images", landmark_image_filename).replace("\\", "/")
                    markdown_lines.append(f"![Foto del Landmark {lm['landmark_id']}]({landmark_image_display_path})\n")
                except Exception as e_img:
                     markdown_lines.append(f"*No se pudo generar el enlace para la foto del landmark {lm['landmark_id']}: {e_img} (Path original: {lm['best_image_path']})*\n")
            else:
                markdown_lines.append(f"*Foto del landmark {lm['landmark_id']} no disponible o path incorrecto ({lm['best_image_path']}).*\n")

            markdown_lines.append(f"- **Nombre/Categoría:** {lm['object_name_or_category']}")
            
            # Manejar descripciones y análisis multilínea
            markdown_lines.append("- **Descripción Visual Detallada:**")
            for line in lm['detailed_visual_description'].split('\n'):
                markdown_lines.append(f"  {line}") # Indentado para claridad como sub-item

            markdown_lines.append("- **Análisis Contextual Marciano:**")
            for line in lm['contextual_analysis'].split('\n'):
                markdown_lines.append(f"  {line}")

            markdown_lines.append("- **Ubicación Estimada (Robot Pose):**")
            markdown_lines.append(f"    - Timestamp: {lm['estimated_location']['timestamp_ms']} ms")
            markdown_lines.append(f"    - X: {lm['estimated_location']['x']:.2f} m, Y: {lm['estimated_location']['y']:.2f} m")
            markdown_lines.append(f"    - Orientación: {lm['estimated_location']['orientation_degrees']:.1f}°")
            
            markdown_lines.append("\n---\n") # Separador entre landmarks

        return markdown_lines

    def generate_markdown_report(self, markdown_content: List[str], mission_id: str) -> str:
        """Genera el reporte Markdown y lo guarda en un archivo .md."""
        report_filename_md = f"ERC2025_Report_{mission_id}.md"
        report_filepath = os.path.join(self.output_dir, report_filename_md)

        try:
            with open(report_filepath, "w", encoding="utf-8") as f:
                for line in markdown_content:
                    f.write(line + "\n") # Asegurar nueva línea para cada elemento de la lista
            print(f"Reporte Markdown generado: {report_filepath}")
            return report_filepath
        except Exception as e:
            print(f"Error al escribir el archivo Markdown: {e}")
            return "" # Retornar string vacío o None en caso de error

    def run(self, identified_landmarks_batch: IdentifiedLandmarksBatchState) -> str:
        """
        Orquesta la preparación del contenido y la generación del archivo Markdown.
        Retorna el path al archivo Markdown generado.
        """
        if not identified_landmarks_batch or not identified_landmarks_batch.get('mission_id'):
            print("Agente Generador de Reportes: Datos de entrada inválidos o mission_id faltante.")
            return "No se pudo generar el reporte debido a datos de entrada inválidos."

        print(f"Agente Generador de Reportes (Markdown): Iniciando para misión {identified_landmarks_batch['mission_id']}...")
        
        markdown_content_list = self._prepare_markdown_content(identified_landmarks_batch)
        report_file_path = self.generate_markdown_report(markdown_content_list, identified_landmarks_batch['mission_id'])
        
        print(f"Agente Generador de Reportes (Markdown): Finalizado.")
        return report_file_path