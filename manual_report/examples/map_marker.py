#!/usr/bin/env python3
"""
Annotate a map image (PGM) using the map.yaml metadata (resolution, origin).
Handles rotated maps (yaw != 0), OpenCV top-left pixel coords, and clamps/out-of-bounds.
"""

import os
import math
import yaml
import cv2
import numpy as np
import json 
def load_map_and_metadata(yaml_path):
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    img_path = meta.get('image')
    # support yaml image path relative to yaml file
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"map image not found: {img_path}")
    # Ensure grayscale to color for annotation
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
    return img, img_color, meta, img_path

def world_to_pixel(wx, wy, meta, img_shape):
    """
    Convert world (wx,wy) in map frame -> (col, row_top) pixel coords for OpenCV.
    Returns (col, row_top) as integers, and a bool in_bounds.
    """
    res = float(meta['resolution'])
    ox, oy, oyaw = meta.get('origin', [0.0, 0.0, 0.0])
    yaw = float(oyaw)

    # vector from origin to world point
    dx = float(wx) - float(ox)
    dy = float(wy) - float(oy)

    cosy = math.cos(yaw)
    siny = math.sin(yaw)

    # inverse rotation: map_frame = R^T * (world - origin)
    x_map =  cosy * dx + siny * dy
    y_map = -siny * dx + cosy * dy

    # pixel indices measured from bottom-left corner:
    col = int(math.floor(x_map / res))
    row_from_bottom = int(math.floor(y_map / res))

    img_h, img_w = img_shape[0], img_shape[1]
    # convert to OpenCV top-left row index
    row_top = img_h - 1 - row_from_bottom

    in_bounds = (0 <= col < img_w) and (0 <= row_top < img_h)
    return col, row_top, in_bounds

def pixel_to_world(col, row_top, meta, img_shape):
    """
    Convert OpenCV pixel coords (col, row_top) -> world (wx,wy).
    Uses pixel CENTER: (col + 0.5, row_from_bottom + 0.5)
    """
    res = float(meta['resolution'])
    ox, oy, oyaw = meta.get('origin', [0.0, 0.0, 0.0])
    yaw = float(oyaw)

    img_h, img_w = img_shape[0], img_shape[1]
    row_from_bottom = img_h - 1 - row_top

    # meters from bottom-left origin to pixel center
    x_map = (col + 0.5) * res
    y_map = (row_from_bottom + 0.5) * res

    cosy = math.cos(yaw)
    siny = math.sin(yaw)

    # world = origin + R * [x_map, y_map]
    wx = ox + cosy * x_map - siny * y_map
    wy = oy + siny * x_map + cosy * y_map
    return wx, wy

def annotate_map(yaml_path, json_path, out_path="annotated_map.png",
                 marker_color=(0,0,255), marker_radius=6, marker_thickness=-1,
                 draw_labels=True):
    img_gray, img_color, meta, img_path = load_map_and_metadata(yaml_path)
    h, w = img_gray.shape[0], img_gray.shape[1]

    # normalize meta origin to 3 elements
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    if len(origin) < 3:
        origin = [float(origin[0]), float(origin[1]), 0.0]
    meta['origin'] = [float(origin[0]), float(origin[1]), float(origin[2])]
    with open(json_path, 'r') as f:
        markers = json.load(f)

    for marker in markers:
        name = marker.get("name", "?")
        wx, wy = float(marker["x"]), float(marker["y"])
        col, row_top, in_bounds = world_to_pixel(wx, wy, meta, img_gray.shape)
        if in_bounds:
            cv2.circle(img_color, (col, row_top), marker_radius, marker_color, marker_thickness)
            if draw_labels:
                label = f"{name}: ({wx:.2f},{wy:.2f})"
                # put text slightly above the circle (ensure within image)
                text_pos = (max(0, col+8), max(12, row_top-8))
                cv2.putText(img_color, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, marker_color, 1, cv2.LINE_AA)
        else:
            print(f"WARNING: world point {name} ({wx},{wy}) -> pixel ({col},{row_top}) is OUT OF IMAGE BOUNDS [{w}x{h}]")

    # Save annotated image
    cv2.imwrite(out_path, img_color)
    print(f"Annotated map written to: {out_path}")
    return out_path

if __name__ == "__main__":
    # Example usage – edit these for your map and coordinates
    yaml_path = "marsyard_savemap.yaml"
    # world coords you want to mark (map frame)
    out = annotate_map(yaml_path, 'markers.json', out_path="annotated_map.png")
