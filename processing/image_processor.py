import cv2
import numpy as np
import os

def generate_visualizations(input_path, output_folder, base_filename):
    """Genera imágenes de diagnóstico visual"""
    # Cargar imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Normalizar
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 1. Imagen original
    original_path = os.path.join(output_folder, f"{base_filename}_original.png")
    cv2.imwrite(original_path, img)
    
    # 2. Contornos
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 1)
    contour_path = os.path.join(output_folder, f"{base_filename}_contours.png")
    cv2.imwrite(contour_path, contour_img)
    
    # 3. Mapa de calor
    blur = cv2.GaussianBlur(img, (5,5), 0)
    heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    heat_path = os.path.join(output_folder, f"{base_filename}_heatmap.png")
    cv2.imwrite(heat_path, heatmap)
    
    return {
        'original': f"results/{base_filename}_original.png",
        'contours': f"results/{base_filename}_contours.png",
        'heatmap': f"results/{base_filename}_heatmap.png"
    }