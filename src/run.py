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

def print_ascii_art():
    """
    Prints ASCII art for "code provided by #Terrabots".
    Font: Big
    """
    green_color = "\033[92m"
    reset_color = "\033[0m"
    art = f"""{green_color}
    Landmarks detection Pipeline ERC 2025
    code provided by
    ----------------------------------------------------------------------------------------------------------------
    ___  ___  _  _  ____    ____  ____  _  _  __  __  ____  _  _  ____  ____  ___  ____
░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░▒▓████████▓▒░▒▓███████▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░  ░▒▓█▓▒░        
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░  ░▒▓█▓▒░        
   ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓███████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓██████▓▒░  
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░         ░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░         ░▒▓█▓▒░ 
   ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░  ░▒▓█▓▒░  ░▒▓███████▓▒░  

    ----------------------------------------------------------------------------------------------------------------
    {reset_color}                                                                                                              
    """
    print(art)

def main(video_path: str, pose_data_path: str, mission_id: str):
    pipeline_start_time = time.time()
    step_times = {}
    
    print_ascii_art()
    print(f"---🚀 Starting landmark detection pipeline: {mission_id} ---")

    base_output_dir = "outputs/output5"
    temp_segment_dir = os.path.join(base_output_dir, "temp_video_segments")
    landmark_image_dir = os.path.join(base_output_dir, "landmark_images")
    report_dir = os.path.join(base_output_dir, "reports")
    map_image_dir = os.path.join(base_output_dir, "map_images")

    os.makedirs(temp_segment_dir, exist_ok=True)
    os.makedirs(landmark_image_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(map_image_dir, exist_ok=True)

    robot_poses = load_sample_robot_poses(pose_data_path)
    mission_input = MissionInputState(
        video_path=video_path,
        robot_poses=robot_poses,
        mission_id=mission_id
    )

    preprocessor = PreprocessorAgent(segment_output_dir=temp_segment_dir)
    analyst = AnalystAgent() 
    identifier = IdentifierAgent(output_landmark_image_dir=landmark_image_dir)
    report_generator = ReportGeneratorAgent(output_dir=report_dir, map_image_dir=map_image_dir)

    print("\n---👁️ Step 1: Video Preprocessing ---")
    step_start = time.time()
    preprocessed_video_segments = preprocessor.run(mission_input)
    step_times['preprocessing'] = time.time() - step_start
    if not preprocessed_video_segments:
        print("Segments from Videos were not found, killing process...")
        return

    print(f"\n---🤖 Step 2: Video Analysis with Gemini ---")
    step_start = time.time()
    analyzed_segments = analyst.run(preprocessed_video_segments)
    step_times['analysis'] = time.time() - step_start
    if not analyzed_segments: 
        print("Video Analysis has failed. Review AnalystAgents logs")
    
    found_any_observations = any(seg["identified_landmark_observations"] for seg in analyzed_segments)
    if not found_any_observations and analyzed_segments: 
        print("AnalystAgent completed, but no landmark observations were found in the segments.")

    print("\n---🧠 Step 3: Landmarks identification and contextualization ---")
    step_start = time.time()
    all_poses_for_map = mission_input['robot_poses']
    identified_batch = identifier.run(analyzed_segments, all_poses_for_map)
    step_times['identification'] = time.time() - step_start
    
    print("\n---📖 Step 4: Report Generation ---")
    step_start = time.time()
    report_file_path = report_generator.run(identified_batch)
    step_times['report_generation'] = time.time() - step_start

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print(f"\n---✅ COMPLETED PIPELINE: {mission_id} ---")
    print(f"Generated report: {report_file_path}")
    print("\nStep-by-step timing:")
    print(f"Step 1 - Preprocessing: {step_times['preprocessing']:.2f} seconds")
    print(f"Step 2 - Analysis: {step_times['analysis']:.2f} seconds")
    print(f"Step 3 - Identification: {step_times['identification']:.2f} seconds")
    print(f"Step 4 - Report Generation: {step_times['report_generation']:.2f} seconds")
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmarks detection Pipeline ERC 2025")
    parser.add_argument("video_path", help="Video path")
    parser.add_argument("--pose_file", default=None, help="(optional) path to trajectory file in json ")
    parser.add_argument("--mission_id", default=f"mission_{int(time.time())}", help="id")    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: video in '{args.video_path}' was not found.")
    else:
        main(args.video_path, args.pose_file, args.mission_id)