import os
import time
from PIL import Image, ImageDraw
import numpy as np

# Importar las funciones necesarias desde el generador de reportes
from report_generator import (
    generate_markdown_report,
    save_report_to_file,
    convert_md_to_pdf,
    generate_mission_map,
    read_trajectory_data # <-- 1. IMPORTAR LA NUEVA FUNCIÓN DE LECTURA
)

# --- Configuración ---
TEST_ASSETS_DIR = "test_temp_assets"

def create_dummy_image(filepath: str):
    """Crea una imagen de marcador de posición simple para las pruebas."""
    try:
        img_size = (400, 300)
        img = Image.new('RGB', img_size, color = (210, 210, 210))
        d = ImageDraw.Draw(img)
        d.text((50,130), "Imagen de prueba para landmark", fill=(0,0,0))
        img.save(filepath)
        print(f"Imagen de prueba creada: {filepath}")
    except Exception as e:
        print(f"No se pudo crear la imagen de prueba. Asegúrate de tener 'Pillow' instalado (`pip install Pillow`). Error: {e}")
        raise

def create_dummy_trajectory_file(filepath: str):
    """Crea un archivo de trayectoria falso con una ruta sinusoidal."""
    print(f"Creando archivo de trayectoria de prueba en: {filepath}")
    try:
        with open(filepath, "w") as f:
            # Generar una trayectoria simple en forma de curva
            x = np.linspace(100, 130, 50)
            y = 5 * np.sin(np.linspace(0, 2 * np.pi, 50)) - 40
            for i in range(len(x)):
                f.write(f"{x[i]:.4f},{y[i]:.4f}\n")
    except Exception as e:
        print(f"No se pudo crear el archivo de trayectoria de prueba. Error: {e}")
        raise

def get_hardcoded_landmarks(dummy_image_path: str) -> list:
    """Proporciona una lista estática de datos de landmarks para pruebas consistentes."""
    return [
        {
            'id': 'LM_TEST_001',
            'name': 'Ensamblaje de Cabeza de Taladro',
            'timestamp': 1724526000,
            'location': {'x': 105.45, 'y': -30.12, 'z': 0.55},
            'best_image_path': dummy_image_path,
            'detailed_description': "Un objeto metálico con forma de cono parcialmente enterrado en el regolito marciano.",
            'contextual_analysis': "Origen probable: Cabeza de taladro de una misión geológica anterior."
        },
        {
            'id': 'LM_TEST_002',
            'name': 'Fragmento de Panel de Control',
            'timestamp': 1724529800,
            'location': {'x': 112.80, 'y': -45.60, 'z': 0.90},
            'best_image_path': "imagen_inexistente.jpg", # Prueba para una imagen faltante
            'detailed_description': "Una placa plana y rectangular con varios botones descoloridos.",
            'contextual_analysis': "Relevancia/Importancia: Media. Ayuda a reconstruir eventos."
        },
        {
            'id': 'LM_TEST_003',
            'name': 'Objeto Esférico No Identificado',
            'timestamp': 1724531000,
            'location': {'x': 120.10, 'y': -50.20, 'z': 1.10},
            'best_image_path': dummy_image_path,
            'detailed_description': "Un objeto perfectamente esférico con una superficie lisa y no reflectante.",
            'contextual_analysis': "Relevancia/Importancia: Muy Alta. Podría ser un hallazgo geológico significativo."
        }
    ]

def main():
    """Función principal para ejecutar la prueba de formato de PDF."""
    print("--- Iniciando Prueba de Formato de PDF ---")
    
    if not os.path.exists(TEST_ASSETS_DIR):
        os.makedirs(TEST_ASSETS_DIR)

    test_run_id = f"test_{int(time.time())}"
    dummy_image = os.path.join(TEST_ASSETS_DIR, f"dummy_image_{test_run_id}.png")
    dummy_map = os.path.join(TEST_ASSETS_DIR, f"dummy_map_{test_run_id}.png")
    dummy_trajectory = os.path.join(TEST_ASSETS_DIR, "dummy_path.txt") # <-- 2. DEFINIR RUTA PARA TRAYECTORIA
    md_filepath = ""

    try:
        # 1. Crear activos de prueba: imagen y archivo de trayectoria
        create_dummy_image(dummy_image)
        create_dummy_trajectory_file(dummy_trajectory) # <-- 3. CREAR EL ARCHIVO DE TRAYECTORIA

        # 2. Obtener los datos de landmarks y leer la trayectoria
        landmarks_data = get_hardcoded_landmarks(dummy_image)
        path_x, path_y = read_trajectory_data(dummy_trajectory) # <-- 4. LEER LA TRAYECTORIA

        # 3. Generar el mapa de la misión con landmarks y trayectoria
        generate_mission_map(landmarks_data, path_x, path_y, dummy_map) # <-- 5. PASAR DATOS DEL MAPA

        # 4. Generar el contenido del informe en Markdown
        report_content = generate_markdown_report(landmarks_data, dummy_map)

        # 5. Guardar el informe Markdown en un archivo
        mission_id = "PDF_FORMAT_TEST"
        md_filepath = save_report_to_file(report_content, mission_id)

        # 6. Convertir el archivo Markdown a PDF
        if md_filepath:
            pdf_filepath = md_filepath.replace(".md", ".pdf")
            convert_md_to_pdf(md_filepath, pdf_filepath)
        else:
            print("❌ No se pudo crear el archivo Markdown, se omite la conversión a PDF.")

    except Exception as e:
        print(f"\nOcurrió un error durante la ejecución de la prueba: {e}")
        
if __name__ == "__main__":
    # Nota: este script requiere 'Pillow' y 'matplotlib'
    # Instálalos usando: pip install Pillow matplotlib numpy
    main()
