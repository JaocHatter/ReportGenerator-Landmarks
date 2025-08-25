import os
import time
from PIL import Image, ImageDraw
import numpy as np
import shutil

# Importar la clase refactorizada desde el mismo directorio de la app
from report_generator import ReportGenerator

# --- Configuración ---
TEST_OUTPUT_DIR = "test_temp_output"

def create_dummy_image(filepath: str):
    """Crea una imagen de marcador de posición simple para las pruebas."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        img = Image.new('RGB', (400, 300), color='lightgrey')
        d = ImageDraw.Draw(img)
        d.text((50, 130), "Landmark Test Image", fill='black')
        img.save(filepath)
        print(f"Imagen de prueba creada: {filepath}")
    except Exception as e:
        print(f"No se pudo crear la imagen de prueba: {e}")
        raise

def create_dummy_trajectory_file(filepath: str):
    """Crea un archivo de trayectoria falso."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Creando archivo de trayectoria de prueba en: {filepath}")
    try:
        with open(filepath, "w") as f:
            x = np.linspace(100, 130, 50)
            y = 5 * np.sin(np.linspace(0, 2 * np.pi, 50)) - 40
            for i in range(len(x)):
                f.write(f"{x[i]:.4f},{y[i]:.4f}\n")
    except Exception as e:
        print(f"No se pudo crear el archivo de trayectoria de prueba: {e}")
        raise

def get_hardcoded_landmarks(dummy_image_path: str) -> list:
    """Proporciona una lista estática de datos de landmarks para pruebas consistentes."""
    return [
        {'id': 'LM_TEST_001', 'name': 'Drill Head Assembly', 'timestamp': 1724526000, 'location': {'x': 105.45, 'y': -30.12, 'z': 0.55}, 'best_image_path': dummy_image_path, 'detailed_description': "A metallic, cone-shaped object.", 'contextual_analysis': "Probable origin: Prior geological mission."},
        {'id': 'LM_TEST_002', 'name': 'Control Panel Fragment', 'timestamp': 1724529800, 'location': {'x': 112.80, 'y': -45.60, 'z': 0.90}, 'best_image_path': "non_existent_image.jpg", 'detailed_description': "A flat, rectangular plate.", 'contextual_analysis': "Relevance: Medium."},
        {'id': 'LM_TEST_003', 'name': 'Unidentified Spherical Object', 'timestamp': 1724531000, 'location': {'x': 120.10, 'y': -50.20, 'z': 1.10}, 'best_image_path': dummy_image_path, 'detailed_description': "A perfectly spherical object.", 'contextual_analysis': "Relevance: Very High."}
    ]

def main():
    """Función principal para ejecutar la prueba de formato de PDF."""
    print("--- Iniciando Prueba de Simulación de Reporte ---")
    
    # Definir rutas para los archivos de prueba
    dummy_image_path = os.path.join(TEST_OUTPUT_DIR, "images", "dummy_image.png")
    dummy_trajectory_path = os.path.join(TEST_OUTPUT_DIR, "trajectory", "path.txt")

    try:
        # 1. Crear activos de prueba
        create_dummy_image(dummy_image_path)
        create_dummy_trajectory_file(dummy_trajectory_path)

        # 2. Obtener los datos de landmarks
        landmarks_data = get_hardcoded_landmarks(dummy_image_path)

        # 3. Instanciar y ejecutar el generador de reportes
        print("\n--- Generando Reporte ---")
        report_generator = ReportGenerator(
            landmarks_data=landmarks_data,
            trajectory_data_path=dummy_trajectory_path
        )
        pdf_filepath = report_generator.generate_report()
        
        print(f"\n✅ Proceso de prueba completado. Reporte generado en: {pdf_filepath}")

    except Exception as e:
        print(f"\n❌ Ocurrió un error durante la ejecución de la prueba: {e}")

if __name__ == "__main__":
    # Nota: este script requiere 'Pillow', 'matplotlib' y 'numpy'
    # Instálalos usando: pip install Pillow matplotlib numpy
    main()
