import time
import argparse
import json 
from states import MissionInputState, RobotPose
from agents import PreprocessorAgent, AnalystAgent, IdentifierAgent, ReportGeneratorAgent
import os

def load_sample_robot_poses(pose_file_path: str = None) -> list[RobotPose]:
    """Carga poses de un archivo JSON o devuelve datos de ejemplo."""
    if pose_file_path:
        try:
            with open(pose_file_path, 'r') as f:
                poses_data = json.load(f)
                return [RobotPose(**p) for p in poses_data]
        except Exception as e:
            print(f"Error cargando archivo de poses {pose_file_path}: {e}. Usando datos de ejemplo.")

    return [
        RobotPose(timestamp_ms=0, x=0.0, y=0.0, orientation_degrees=0.0),
        RobotPose(timestamp_ms=1000, x=1.0, y=0.1, orientation_degrees=5.0),
        RobotPose(timestamp_ms=2000, x=2.0, y=0.3, orientation_degrees=10.0),
        RobotPose(timestamp_ms=3000, x=3.0, y=0.4, orientation_degrees=15.0),
        RobotPose(timestamp_ms=4000, x=4.0, y=0.5, orientation_degrees=20.0),
        RobotPose(timestamp_ms=5000, x=5.0, y=0.5, orientation_degrees=20.0),
    ]

def main(video_path: str, pose_data_path: str, mission_id: str):
    start_time = time.time()
    print(f"--- Iniciando Pipeline de Detección de Landmarks para Misión: {mission_id} ---")

    robot_poses = load_sample_robot_poses(pose_data_path)
    mission_input = MissionInputState(
        video_path=video_path,
        robot_poses=robot_poses,
        mission_id=mission_id
    )

    preprocessor = PreprocessorAgent(output_dir="output/temp_frames")
    analyst = AnalystAgent()
    identifier = IdentifierAgent(output_landmark_image_dir="output/landmark_images")
    report_generator = ReportGeneratorAgent(output_dir="output/reports", map_image_dir="output/map_images")

    base_output_dir = "output"
    temp_segment_dir = os.path.join(base_output_dir, "temp_video_segments")
    landmark_image_dir = os.path.join(base_output_dir, "landmark_images")
    report_dir = os.path.join(base_output_dir, "reports")
    map_image_dir = os.path.join(base_output_dir, "map_images")

    os.makedirs(temp_segment_dir, exist_ok=True)
    os.makedirs(landmark_image_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(map_image_dir, exist_ok=True)

    # 1. Cargar datos de entrada (sin cambios)
    robot_poses = load_sample_robot_poses(pose_data_path)
    mission_input = MissionInputState(
        video_path=video_path,
        robot_poses=robot_poses,
        mission_id=mission_id
    )

    # 2. Instanciar Agentes (actualizados)
    preprocessor = PreprocessorAgent(segment_output_dir=temp_segment_dir)
    analyst = AnalystAgent() # Ya no necesita output_dir para frames
    identifier = IdentifierAgent(output_landmark_image_dir=landmark_image_dir)
    report_generator = ReportGeneratorAgent(output_dir=report_dir, map_image_dir=map_image_dir)


    # 3. Ejecutar la cadena de agentes (con los nuevos estados)
    print("\n--- Etapa 1: Preprocesamiento de Video/Segmentos ---")
    # Ahora retorna List[PreprocessedVideoSegmentState]
    preprocessed_video_segments = preprocessor.run(mission_input)
    if not preprocessed_video_segments:
        print("No se generaron segmentos de video preprocesados. Terminando.")
        return

    print(f"\n--- Etapa 2: Análisis de Video con Gemini (Puede tardar MUCHO) ---")
    # Ahora retorna List[AnalyzedVideoSegmentState]
    analyzed_segments = analyst.run(preprocessed_video_segments)
    if not analyzed_segments: # Podría ser una lista vacía si no hay nada o error
        print("El análisis de video no produjo resultados o falló. Revisar logs del AnalystAgent.")
    
    found_any_observations = any(seg.identified_landmark_observations for seg in analyzed_segments)
    if not found_any_observations and analyzed_segments: # Hay segmentos analizados, pero sin observaciones
        print("AnalystAgent completado, pero no se encontraron observaciones de landmarks en los segmentos.")

    print("\n--- Etapa 3: Identificación y Contextualización de Landmarks ---")
    all_poses_for_map = mission_input['robot_poses']
    # Identifier ahora toma List[AnalyzedVideoSegmentState]
    identified_batch = identifier.run(analyzed_segments, all_poses_for_map)

    print("\n--- Etapa 4: Generación de Reporte ---")
    report_file_path = report_generator.run(identified_batch)

    end_time = time.time()
    print(f"\n--- Pipeline Completado para Misión: {mission_id} ---")
    print(f"Reporte generado en: {report_file_path}")
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Detección de Landmarks ERC 2025.")
    parser.add_argument("video_path", help="Ruta al archivo de video de la misión.")
    parser.add_argument("--pose_file", default=None, help="(Opcional) Ruta al archivo JSON con datos de pose del robot.")
    parser.add_argument("--mission_id", default=f"mission_{int(time.time())}", help="ID único para esta corrida de la misión.")    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: El archivo de video '{args.video_path}' no fue encontrado.")
    else:
        main(args.video_path, args.pose_file, args.mission_id)