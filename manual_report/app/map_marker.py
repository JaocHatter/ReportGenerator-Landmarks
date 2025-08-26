import os
import math
import yaml
import cv2
import numpy as np
import json

class MapAnnotator:
    def __init__(self, yaml_path, pgm_path):
        with open(yaml_path, 'r') as f:
            self.meta = yaml.safe_load(f)

        # support yaml image path relative to yaml file
        if not os.path.isabs(pgm_path):
            pgm_path = os.path.join(os.path.dirname(yaml_path), pgm_path)

        img = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"map image not found: {pgm_path}")

        # Ensure grayscale to color for annotation
        if len(img.shape) == 2:
            self.img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self.img_color = img.copy()
        
        self.img_shape = self.img_color.shape
        
        # normalize meta origin to 3 elements
        origin = self.meta.get('origin', [0.0, 0.0, 0.0])
        if len(origin) < 3:
            origin = [float(origin[0]), float(origin[1]), 0.0]
        self.meta['origin'] = [float(origin[0]), float(origin[1]), float(origin[2])]

    def world_to_pixel(self, wx, wy):
        """
        Convert world (wx,wy) in map frame -> (col, row_top) pixel coords for OpenCV.
        Returns (col, row_top) as integers, and a bool in_bounds.
        """
        res = float(self.meta['resolution'])
        ox, oy, oyaw = self.meta.get('origin', [0.0, 0.0, 0.0])
        yaw = float(oyaw)

        # vector from origin to world point
        dx = float(wx) - float(ox)
        dy = float(wy) - float(oy)

        cosy = math.cos(yaw)
        siny = math.sin(yaw)

        # inverse rotation: map_frame = R^T * (world - origin)
        x_map = cosy * dx + siny * dy
        y_map = -siny * dx + cosy * dy

        # pixel indices measured from bottom-left corner:
        col = int(math.floor(x_map / res))
        row_from_bottom = int(math.floor(y_map / res))

        img_h, img_w = self.img_shape[0], self.img_shape[1]
        # convert to OpenCV top-left row index
        row_top = img_h - 1 - row_from_bottom

        in_bounds = (0 <= col < img_w) and (0 <= row_top < img_h)
        return col, row_top, in_bounds

    def draw_trajectory(self, trajectory_path, color=(0, 255, 0), thickness=1):
        """Dibuja la trayectoria desde un archivo de texto."""
        try:
            with open(trajectory_path, 'r') as f:
                points = []
                for line in f:
                    x_str, y_str = line.strip().split(',')
                    wx, wy = float(x_str), float(y_str)
                    col, row, in_bounds = self.world_to_pixel(wx, wy)
                    if in_bounds:
                        points.append((col, row))
                
                if len(points) > 1:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(self.img_color, [pts], isClosed=False, color=color, thickness=thickness)
        except Exception as e:
            print(f"Error al dibujar la trayectoria: {e}")

    def draw_markers(self, markers_path, color=(0, 0, 255), radius=5, thickness=-1, draw_labels=True):
        """Dibuja los marcadores desde un archivo JSON."""
        try:
            with open(markers_path, 'r') as f:
                markers = json.load(f)
            
            for marker in markers:
                name = marker.get("name", "?")
                wx, wy = float(marker["x"]), float(marker["y"])
                col, row, in_bounds = self.world_to_pixel(wx, wy)

                if in_bounds:
                    cv2.circle(self.img_color, (col, row), radius, color, thickness)
                    if draw_labels:
                        label = f"{name}"
                        text_pos = (max(0, col + 8), max(12, row - 8))
                        cv2.putText(self.img_color, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                else:
                    print(f"Advertencia: El punto {name} ({wx},{wy}) está fuera de los límites del mapa.")
        except Exception as e:
            print(f"Error al dibujar los marcadores: {e}")

    def save_annotated_map(self, out_path):
        """Guarda el mapa anotado."""
        cv2.imwrite(out_path, self.img_color)
        print(f"Mapa anotado guardado en: {out_path}")

