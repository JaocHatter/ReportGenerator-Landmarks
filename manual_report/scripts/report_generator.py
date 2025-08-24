import requests
import time
import os
from typing import List, Dict, Any, Tuple
import markdown2
from weasyprint import HTML, CSS
import base64
import mimetypes
import matplotlib.pyplot as plt

# --- Configuración ---
API_BASE_URL = "http://127.0.0.1:8000"
LANDMARKS_ENDPOINT = f"{API_BASE_URL}/landmarks/"
REPORTS_DIR = "mission_reports"
ASSETS_DIR = "assets"
# La ruta al archivo de trayectoria que genera la API
TRAJECTORY_FILE_PATH = "output/trajectory_data/path.txt"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
MAP_FILENAME = "temp_mission_overview_map.png"

def fetch_landmarks_from_api() -> List[Dict[str, Any]]:
    """Obtiene la lista completa de landmarks desde nuestra API en ejecución."""
    print(f"Intentando obtener datos de {LANDMARKS_ENDPOINT}...")
    try:
        response = requests.get(LANDMARKS_ENDPOINT)
        response.raise_for_status()
        print("Datos obtenidos correctamente.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: No se pudo conectar a la API. Asegúrate de que el servidor FastAPI esté en ejecución. Detalle: {e}")
        return []

def read_trajectory_data(filepath: str) -> Tuple[List[float], List[float]]:
    """Lee el archivo de datos de trayectoria y devuelve las coordenadas X e Y."""
    path_x, path_y = [], []
    if not os.path.exists(filepath):
        print(f"Advertencia: No se encontró el archivo de trayectoria en {filepath}")
        return path_x, path_y
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    path_x.append(float(parts[0]))
                    path_y.append(float(parts[1]))
    except Exception as e:
        print(f"Error al leer el archivo de trayectoria: {e}")
    return path_x, path_y

def image_to_base64_uri(filepath: str) -> str:
    """Lee un archivo de imagen y lo convierte a un Data URI Base64."""
    try:
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type: mime_type = "application/octet-stream"
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"No se pudo convertir la imagen a Base64: {e}")
        return ""

def generate_mission_map(landmarks: List[Dict[str, Any]], path_x: List[float], path_y: List[float], output_path: str) -> bool:
    """Genera y guarda un gráfico 2D de la ruta del rover y la ubicación de los landmarks."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Graficar la trayectoria del rover si existe
    if path_x and path_y:
        ax.plot(path_x, path_y, color='gray', linestyle='--', linewidth=1.5, label='Trayectoria del Rover')

    # Graficar los landmarks si existen
    if landmarks:
        lm_x = [lm['location']['x'] for lm in landmarks]
        lm_y = [lm['location']['y'] for lm in landmarks]
        labels = [lm['id'] for lm in landmarks]
        ax.scatter(lm_x, lm_y, c='black', marker='X', s=100, label='Landmarks', zorder=5)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (lm_x[i], lm_y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='black')

    ax.set_title('Mapa de Operaciones de la Misión', fontsize=16, color='black')
    ax.set_xlabel('Coordenada X (metros)', color='black')
    ax.set_ylabel('Coordenada Y (metros)', color='black')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    fig.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Mapa de la misión generado: {output_path}")
        return True
    except Exception as e:
        print(f"Error al generar el mapa: {e}")
        return False

def generate_markdown_report(landmarks: List[Dict[str, Any]], map_image_path: str) -> str:
    """Genera un informe de misión completo en Markdown."""
    mission_id = f"MISION_{time.strftime('%Y%m%d')}"
    report_lines = [f"# Informe de Misión ERC 2025: {mission_id}"]
    report_lines.append(f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report_lines.append("\n[TOC]\n")

    report_lines.append(f"\n## Resumen de la Misión")
    report_lines.append(f"- **Total de Landmarks Confirmados:** {len(landmarks)}")
    
    if map_image_path and os.path.exists(map_image_path):
        map_uri = image_to_base64_uri(map_image_path)
        report_lines.append(f'\n### Mapa de Operaciones\n<img src="{map_uri}" alt="Mapa General de la Misión" class="map-image">')

    for lm in landmarks:
        lm_id = lm.get('id', 'N/A')
        report_lines.append(f"\n## Landmark: {lm_id} {{#{lm_id}}}")
        
        image_path = lm.get('best_image_path')
        if image_path and os.path.exists(image_path):
            base64_uri = image_to_base64_uri(image_path)
            if base64_uri:
                report_lines.append(f"\n![Foto de {lm.get('name', 'N/A')}]({base64_uri})\n")
        else:
            report_lines.append(f"\n*Imagen no disponible.*\n")
        
        report_lines.append(f"### Nombre/Categoría\n**{lm.get('name', 'N/A')}**")
        ts = lm.get('timestamp', 0)
        report_lines.append(f"### Timestamp de Observación\n{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(ts))}")
        loc = lm.get('location', {})
        report_lines.append(f"### Ubicación Estimada\n`X={loc.get('x', 0):.2f}m, Y={loc.get('y', 0):.2f}m, Z={loc.get('z', 0):.2f}m`")
        desc = lm.get('detailed_description', 'No proporcionada.').replace('\n', '\n> ')
        report_lines.append(f"### Descripción Visual Detallada\n> {desc}")
        analysis = lm.get('contextual_analysis', 'No proporcionado.').replace('\n', '\n> ')
        report_lines.append(f"### Análisis Contextual Marciano\n> {analysis}")
        
    return "\n".join(report_lines)
        
def convert_md_to_pdf(markdown_filepath: str, pdf_filepath: str):
    """Convierte un archivo Markdown a PDF con estilos avanzados."""
    print(f"Iniciando conversión a PDF para: {markdown_filepath}")
    try:
        with open(markdown_filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()

        logo_uri = image_to_base64_uri(LOGO_PATH) if os.path.exists(LOGO_PATH) else ""

        css_style = f"""
        @page {{
            size: A4; margin: 1in;
            @top-left {{ content: 'Informe de Misión ERC 2025'; font-size: 9pt; color: #888; }}
            @top-right {{ content: url('{logo_uri}'); transform: scale(0.4); position: absolute; top: -20px; right: 0; }}
            @bottom-center {{ content: "Página " counter(page) " de " counter(pages); font-size: 9pt; color: #888; }}
        }}
        body {{ font-family: 'Helvetica', sans-serif; font-size: 11pt; line-height: 1.4; }}
        h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 25px; }}
        h2 {{ page-break-before: always; border-bottom: 1px solid #ccc; padding-top: 15px; font-size: 16pt; }}
        h3 {{ font-size: 12pt; font-weight: bold; margin-bottom: -5px; }}
        img {{ display: block; margin: 20px auto; max-width: 60%; max-height: 35vh; border: 1px solid #ddd; padding: 4px; }}
        .map-image {{ max-width: 95%; max-height: none; }}
        blockquote {{ margin-left: 15px; padding-left: 15px; border-left: 3px solid #eee; font-style: italic; color: #333; }}
        .toc {{ border: 1px solid #ccc; background-color: #f9f9f9; padding: 15px; margin-bottom: 25px; page-break-after: always; }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc a {{ text-decoration: none; color: #337ab7; }}
        """
        
        html_body = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'toc', 'markdown-in-html'])
        full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_body}</body></html>"
        
        HTML(string=full_html).write_pdf(pdf_filepath, stylesheets=[CSS(string=css_style)])
        print(f"✅ Informe PDF generado correctamente: {pdf_filepath}")
    except Exception as e:
        print(f"❌ Error al generar el PDF: {e}")

def save_report_to_file(report_content: str, mission_id: str) -> str:
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    filename = f"ERC2025_Informe_{mission_id}_{int(time.time())}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n✅ Informe Markdown generado correctamente: {filepath}")
    return filepath

def main():
    """Función principal para obtener datos y generar informes PDF mejorados."""
    landmarks = fetch_landmarks_from_api()
    path_x, path_y = read_trajectory_data(TRAJECTORY_FILE_PATH)

    if not landmarks and not path_x:
        print("No se encontraron landmarks ni datos de trayectoria. No se generará el informe.")
        return

    map_path = os.path.join(REPORTS_DIR, MAP_FILENAME)
    try:
        generate_mission_map(landmarks, path_x, path_y, map_path)
        mission_id = f"MISION_{time.strftime('%Y%m%d')}"
        report_content = generate_markdown_report(landmarks, map_path)
        md_filepath = save_report_to_file(report_content, mission_id)
        
        if md_filepath:
            pdf_filepath = md_filepath.replace(".md", ".pdf")
            convert_md_to_pdf(md_filepath, pdf_filepath)
    finally:
        if os.path.exists(map_path):
            os.remove(map_path)
            print(f"Archivo de mapa temporal eliminado: {map_path}")

if __name__ == "__main__":
    main()
