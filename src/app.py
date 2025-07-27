import subprocess
import sys
import os

def main():
    run_script_path = os.path.join(os.path.dirname(__file__), "run.py")
    command = [sys.executable, run_script_path] + sys.argv[1:]
    
    try:
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing run.py: {e}")
    except FileNotFoundError:
        print(f"Error: No se encontró el script {run_script_path}. Asegúrate que está en el mismo directorio.")

if __name__ == "__main__":
    # Ejemplo de cómo ejecutarlo:
    # python app.py /home/jaoc/Desktop/Projects/ERC/videos/erc_video_test.mp4 --mission_id test001
    # python app.py /home/jaoc/Desktop/Projects/ERC/videos/erc_video_test.mp4 --pose_file path/to/poses.json --mission_id test002
    main()