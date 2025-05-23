#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Programa interactivo para convertir cualquier imagen en un puzzle LEGO
VersiÃ³n optimizada que mejora el agrupamiento de piezas grandes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, color, morphology, feature
import json
from collections import Counter, defaultdict

class LegoPuzzleConverter:
    """
    Clase para convertir una imagen en un puzzle de LEGO
    con mejor preservaciÃ³n de la forma original y optimizaciÃ³n de piezas
    """
    
    # Dimensiones bÃ¡sicas de LEGO en mm
    STUD_WIDTH = 8.0  # Ancho de un stud/punto (P)
    BRICK_HEIGHT = 9.6  # Altura de un brick/ladrillo (H)
    PLATE_HEIGHT = 3.2  # Altura de una plate/placa (h)
    
    def __init__(self, image_path, target_height_cm=50):
        """
        Inicializa el conversor con la imagen y la altura objetivo
        
        Args:
            image_path (str): Ruta a la imagen a convertir
            target_height_cm (float): Altura objetivo del puzzle en cm
        """
        self.image_path = image_path
        self.target_height_cm = target_height_cm
        
        # Cargar la imagen
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR a RGB (OpenCV carga en BGR)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Obtener dimensiones originales
        self.original_height, self.original_width = self.original_image.shape[:2]
        
        # Calcular dimensiones del puzzle en studs
        self.target_height_mm = target_height_cm * 10  # Convertir cm a mm
        self.studs_height = int(self.target_height_mm / self.STUD_WIDTH)
        self.studs_width = int(self.studs_height * (self.original_width / self.original_height))
        
        # Redimensionar la imagen al tamaÃ±o en studs
        self.resized_image = cv2.resize(self.original_image, (self.studs_width, self.studs_height), 
                                        interpolation=cv2.INTER_AREA)
        
        # Inicializar variables para los siguientes pasos
        self.simplified_image = None
        self.lego_color_image = None
        self.regions = None
        self.basic_shapes = None
        self.lego_pieces = None
        self.piece_cost = 0.0
        self.total_pieces = 0
        
        # Cargar la paleta de colores LEGO
        self.load_lego_colors()
        
        # Cargar informaciÃ³n de piezas LEGO
        self.load_lego_pieces()
        
        # Almacenar el mapeo de colores para la visualizaciÃ³n final
        self.color_mapping = {}
        
        # Matriz para optimizaciÃ³n de piezas
        self.piece_matrix = None
        self.optimized_pieces = None
        
    def load_lego_colors(self):
        """
        Carga la paleta de colores LEGO desde los datos extraÃ­dos
        """
        # Definir colores LEGO bÃ¡sicos (RGB)
        # Basado en la informaciÃ³n de la base de datos proporcionada
        self.lego_colors = {
            # Familia Gris (Greyscale)
            'White': (255, 255, 255),
            'Light Bluish Gray': (175, 181, 199),
            'Dark Bluish Gray': (99, 95, 98),
            'Black': (0, 0, 0),
            
            # Familia Rojo (Red)
            'Red': (200, 0, 0),
            'Dark Red': (137, 0, 0),
            'Coral': (255, 112, 92),
            
            # Familia Naranja (Orange)
            'Orange': (255, 127, 0),
            'Dark Orange': (168, 61, 21),
            'Medium Nougat': (170, 125, 85),
            'Bright Light Orange': (252, 172, 0),
            
            # Familia Amarillo (Yellow)
            'Yellow': (255, 205, 0),
            'Bright Light Yellow': (255, 236, 108),
            
            # Familia Lima (Lime)
            'Lime': (130, 220, 0),
            'Green': (0, 133, 43),
            'Dark Green': (0, 70, 0),
            'Yellowish Green': (192, 255, 0),
            'Bright Green': (0, 200, 0),
            
            # Familia Azul (Blue)
            'Medium Blue': (0, 136, 204),
            'Blue': (0, 87, 168),
            'Dark Blue': (0, 33, 113),
            'Light Blue': (13, 105, 171),
            'Bright Light Blue': (159, 195, 233),
            
            # Otros colores importantes
            'Brown': (88, 42, 18),
            'Reddish Brown': (88, 42, 18),
            'Dark Brown': (53, 33, 0),
            'Tan': (180, 132, 85),
            'Dark Tan': (106, 81, 48),
            'Pink': (220, 144, 149),
            'Magenta': (156, 0, 107),
            'Purple': (52, 0, 128),
        }
        
    def load_lego_pieces(self):
        """
        Carga informaciÃ³n sobre las piezas LEGO disponibles
        """
        # Definir piezas LEGO bÃ¡sicas (ancho x alto en studs)
        self.lego_bricks = [
            # Bricks 1x
            {"name": "Brick 1x1", "id": "3005", "width": 1, "height": 1, "type": "brick"},
            {"name": "Brick 1x2", "id": "3004", "width": 2, "height": 1, "type": "brick"},
            {"name": "Brick 1x3", "id": "3622", "width": 3, "height": 1, "type": "brick"},
            {"name": "Brick 1x4", "id": "3010", "width": 4, "height": 1, "type": "brick"},
            {"name": "Brick 1x6", "id": "3009", "width": 6, "height": 1, "type": "brick"},
            {"name": "Brick 1x8", "id": "3008", "width": 8, "height": 1, "type": "brick"},
            
            # Bricks 2x
            {"name": "Brick 2x2", "id": "3003", "width": 2, "height": 2, "type": "brick"},
            {"name": "Brick 2x3", "id": "3002", "width": 3, "height": 2, "type": "brick"},
            {"name": "Brick 2x4", "id": "3001", "width": 4, "height": 2, "type": "brick"},
            {"name": "Brick 2x6", "id": "2456", "width": 6, "height": 2, "type": "brick"},
            {"name": "Brick 2x8", "id": "3007", "width": 8, "height": 2, "type": "brick"},
            
            # Plates 1x
            {"name": "Plate 1x1", "id": "3024", "width": 1, "height": 1, "type": "plate"},
            {"name": "Plate 1x2", "id": "3023", "width": 2, "height": 1, "type": "plate"},
            {"name": "Plate 1x3", "id": "3623", "width": 3, "height": 1, "type": "plate"},
            {"name": "Plate 1x4", "id": "3710", "width": 4, "height": 1, "type": "plate"},
            {"name": "Plate 1x6", "id": "3666", "width": 6, "height": 1, "type": "plate"},
            {"name": "Plate 1x8", "id": "3460", "width": 8, "height": 1, "type": "plate"},
            
            # Plates 2x
            {"name": "Plate 2x2", "id": "3022", "width": 2, "height": 2, "type": "plate"},
            {"name": "Plate 2x3", "id": "3021", "width": 3, "height": 2, "type": "plate"},
            {"name": "Plate 2x4", "id": "3020", "width": 4, "height": 2, "type": "plate"},
            {"name": "Plate 2x6", "id": "3795", "width": 6, "height": 2, "type": "plate"},
            {"name": "Plate 2x8", "id": "3034", "width": 8, "height": 2, "type": "plate"},
        ]
        
        # Ordenar piezas por tamaÃ±o (de mayor a menor) para optimizaciÃ³n
        self.lego_bricks_by_size = sorted(self.lego_bricks, key=lambda x: x['width'] * x['height'], reverse=True)
    
    def simplify_colors(self, n_colors=32):
        """
        Simplifica los colores de la imagen con mejor preservaciÃ³n de detalles
        
        Args:
            n_colors (int): NÃºmero de colores a los que simplificar
        
        Returns:
            numpy.ndarray: Imagen con colores simplificados
        """
        # Convertir a espacio de color LAB para mejor agrupaciÃ³n perceptual
        lab_image = cv2.cvtColor(self.resized_image, cv2.COLOR_RGB2LAB)
        
        # Aplanar la imagen para k-means
        pixels = lab_image.reshape(-1, 3).astype(np.float32)
        
        # Definir criterios de parada para k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Aplicar k-means para reducir colores
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Convertir los centros a uint8
        centers = np.uint8(centers)
        
        # Reconstruir la imagen con colores simplificados
        simplified_pixels = centers[labels.flatten()]
        simplified_lab = simplified_pixels.reshape(self.resized_image.shape)
        
        # Convertir de LAB a RGB
        self.simplified_image = cv2.cvtColor(simplified_lab, cv2.COLOR_LAB2RGB)
        
        return self.simplified_image
    
    def map_to_lego_colors(self):
        """
        Mapea los colores simplificados a la paleta de colores LEGO
        con mejor preservaciÃ³n de contrastes
        
        Returns:
            numpy.ndarray: Imagen con colores LEGO
        """
        if self.simplified_image is None:
            raise ValueError("Primero debes simplificar los colores de la imagen")
        
        # Crear una imagen para los colores LEGO
        self.lego_color_image = np.zeros_like(self.simplified_image)
        
        # Convertir a espacio HSV para mejor comparaciÃ³n de colores
        hsv_image = cv2.cvtColor(self.simplified_image, cv2.COLOR_RGB2HSV)
        
        # Obtener colores Ãºnicos en la imagen simplificada
        unique_colors = np.unique(self.simplified_image.reshape(-1, 3), axis=0)
        
        # Convertir colores LEGO a HSV para mejor comparaciÃ³n
        lego_hsv = {}
        for name, rgb in self.lego_colors.items():
            rgb_array = np.array([[rgb]], dtype=np.uint8)
            hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            lego_hsv[name] = hsv_array[0, 0]
        
        # Mapear cada color Ãºnico al color LEGO mÃ¡s cercano
        self.color_mapping = {}  # Reiniciar el mapeo de colores
        for color in unique_colors:
            # Convertir a HSV
            color_rgb = np.array([[color]], dtype=np.uint8)
            color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0, 0]
            
            min_distance = float('inf')
            closest_lego_color = None
            closest_lego_name = None
            
            for lego_name, lego_color_hsv in lego_hsv.items():
                # Calcular distancia en espacio HSV con ponderaciÃ³n
                # Dar mÃ¡s importancia al tono (H) y saturaciÃ³n (S) que al valor (V)
                h_weight = 1.0
                s_weight = 0.8
                v_weight = 0.6
                
                # Convertir a float para evitar desbordamiento
                h1 = float(color_hsv[0])
                h2 = float(lego_color_hsv[0])
                s1 = float(color_hsv[1])
                s2 = float(lego_color_hsv[1])
                v1 = float(color_hsv[2])
                v2 = float(lego_color_hsv[2])
                
                # Calcular diferencia de tono (circular)
                h_diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2)) / 180.0
                # Calcular diferencias de saturaciÃ³n y valor
                s_diff = abs(s1 - s2) / 255.0
                v_diff = abs(v1 - v2) / 255.0
                
                distance = h_weight * h_diff + s_weight * s_diff + v_weight * v_diff
                
                if distance < min_distance:
                    min_distance = distance
                    closest_lego_color = self.lego_colors[lego_name]
                    closest_lego_name = lego_name
            
            self.color_mapping[tuple(color)] = (closest_lego_color, closest_lego_name)
        
        # Aplicar el mapeo a la imagen
        for i in range(self.simplified_image.shape[0]):
            for j in range(self.simplified_image.shape[1]):
                color = tuple(self.simplified_image[i, j])
                self.lego_color_image[i, j] = self.color_mapping.get(color, ((0, 0, 0), "Black"))[0]
        
        return self.lego_color_image
    
    def detect_color_regions(self):
        """
        Detecta regiones de colores en la imagen con mejor preservaciÃ³n de formas
        
        Returns:
            dict: Regiones detectadas por color
        """
        if self.lego_color_image is None:
            raise ValueError("Primero debes mapear los colores a la paleta LEGO")
        
        # Crear una mÃ¡scara para el objeto principal
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.lego_color_image, cv2.COLOR_RGB2GRAY)
        
        # Aplicar umbral adaptativo para separar el objeto del fondo
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplicar operaciones morfolÃ³gicas para mejorar la segmentaciÃ³n
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear mÃ¡scara para el objeto principal
        mask = np.zeros_like(gray)
        
        # Seleccionar el contorno mÃ¡s grande (asumiendo que es el objeto principal)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Crear una imagen segmentada usando watershed para mejor separaciÃ³n de regiones
        # Marcadores para watershed
        _, markers = cv2.connectedComponents(binary)
        
        # AÃ±adir 1 a todos los marcadores para asegurar que el fondo es 1, no 0
        markers = markers + 1
        
        # Marcar la regiÃ³n desconocida con 0
        markers[mask == 0] = 0
        
        # Aplicar watershed
        cv2.watershed(self.lego_color_image, markers)
        
        # Crear diccionario para almacenar regiones por color
        self.regions = defaultdict(list)
        
        # Para cada color Ãºnico en la imagen, crear una mÃ¡scara y encontrar regiones
        unique_colors = np.unique(self.lego_color_image.reshape(-1, 3), axis=0)
        
        for color in unique_colors:
            # Crear mÃ¡scara para este color
            color_mask = np.all(self.lego_color_image == color, axis=2).astype(np.uint8) * 255
            
            # Aplicar operaciones morfolÃ³gicas para mejorar la segmentaciÃ³n
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Etiquetar regiones conectadas
            labeled_regions, num_regions = measure.label(color_mask, return_num=True, connectivity=2)
            
            # Para cada regiÃ³n, obtener propiedades
            region_props = measure.regionprops(labeled_regions)
            
            for prop in region_props:
                # Obtener coordenadas de la regiÃ³n
                coords = prop.coords
                
                # Si la regiÃ³n es muy pequeÃ±a, ignorarla
                if len(coords) < 2:
                    continue
                
                # Almacenar la regiÃ³n con su color
                self.regions[tuple(color)].append({
                    'coords': coords,
                    'area': prop.area,
                    'bbox': prop.bbox,  # (min_row, min_col, max_row, max_col)
                    'centroid': prop.centroid,
                    'perimeter': prop.perimeter,
                    'eccentricity': prop.eccentricity,
                    'is_detail': self.is_detail_region(prop, color)  # Identificar si es una regiÃ³n de detalle
                })
        
        return self.regions
    
    def is_detail_region(self, region_prop, color):
        """
        Determina si una regiÃ³n es un detalle importante
        
        Args:
            region_prop: Propiedades de la regiÃ³n
            color: Color de la regiÃ³n
        
        Returns:
            bool: True si es una regiÃ³n de detalle
        """
        # Verificar si es una regiÃ³n pequeÃ±a
        if region_prop.area < 20:
            return True
        
        # Verificar si es una regiÃ³n alargada
        if region_prop.eccentricity > 0.8:
            return True
        
        # Verificar si es una regiÃ³n verde (probablemente vegetaciÃ³n o detalles)
        color_name = self.get_color_name(color)
        if 'Green' in color_name or 'Lime' in color_name or 'Yellowish' in color_name:
            return True
        
        # Verificar si estÃ¡ en los bordes de la imagen
        min_row, min_col, max_row, max_col = region_prop.bbox
        if (min_row < self.studs_height * 0.1 or 
            max_row > self.studs_height * 0.9 or 
            min_col < self.studs_width * 0.1 or 
            max_col > self.studs_width * 0.9):
            return True
        
        return False
    
    def transform_to_basic_shapes(self):
        """
        Transforma las regiones detectadas en formas bÃ¡sicas
        con mejor preservaciÃ³n de contornos
        
        Returns:
            dict: Formas bÃ¡sicas por color
        """
        if self.regions is None:
            raise ValueError("Primero debes detectar las regiones de colores")
        
        self.basic_shapes = defaultdict(list)
        
        for color, regions in self.regions.items():
            for region in regions:
                # Obtener el rectÃ¡ngulo delimitador
                min_row, min_col, max_row, max_col = region['bbox']
                width = max_col - min_col
                height = max_row - min_row
                
                # Si la regiÃ³n es muy pequeÃ±a, representarla como un cuadrado 1x1
                if width < 2 and height < 2:
                    self.basic_shapes[color].append({
                        'type': 'rectangle',
                        'x': min_col,
                        'y': min_row,
                        'width': 1,
                        'height': 1,
                        'is_detail': True  # Las regiones pequeÃ±as son detalles
                    })
                    continue
                
                # Para regiones mÃ¡s grandes, intentar ajustar mejor a la forma
                # Obtener la mÃ¡scara de la regiÃ³n
                mask = np.zeros((self.studs_height, self.studs_width), dtype=np.uint8)
                for y, x in region['coords']:
                    mask[y, x] = 1
                
                # Usar descomposiciÃ³n en rectÃ¡ngulos para cubrir la regiÃ³n
                rectangles = self.decompose_region_to_rectangles(mask, min_row, min_col, max_row, max_col)
                
                # AÃ±adir los rectÃ¡ngulos a las formas bÃ¡sicas
                for rect in rectangles:
                    self.basic_shapes[color].append({
                        'type': 'rectangle',
                        'x': rect[0],
                        'y': rect[1],
                        'width': rect[2],
                        'height': rect[3],
                        'is_detail': region.get('is_detail', False) or rect[2] * rect[3] < 9  # Ãreas pequeÃ±as son detalles
                    })
        
        return self.basic_shapes
    
    def decompose_region_to_rectangles(self, mask, min_row, min_col, max_row, max_col):
        """
        Descompone una regiÃ³n en rectÃ¡ngulos para mejor ajuste a la forma original
        
        Args:
            mask (numpy.ndarray): MÃ¡scara binaria de la regiÃ³n
            min_row, min_col, max_row, max_col: LÃ­mites del rectÃ¡ngulo delimitador
        
        Returns:
            list: Lista de rectÃ¡ngulos (x, y, width, height)
        """
        # Extraer la regiÃ³n de interÃ©s de la mÃ¡scara
        roi = mask[min_row:max_row, min_col:max_col]
        height, width = roi.shape
        
        # Lista para almacenar los rectÃ¡ngulos
        rectangles = []
        
        # Si la regiÃ³n es pequeÃ±a, usar un solo rectÃ¡ngulo
        if width <= 4 and height <= 4:
            # Calcular el Ã¡rea cubierta por la mÃ¡scara
            coverage = np.sum(roi) / (width * height)
            
            # Si la cobertura es alta, usar un rectÃ¡ngulo que cubra toda la regiÃ³n
            if coverage > 0.7:
                rectangles.append((min_col, min_row, width, height))
            else:
                # Dividir en rectÃ¡ngulos mÃ¡s pequeÃ±os
                for y in range(height):
                    for x in range(width):
                        if roi[y, x] == 1:
                            # Buscar el rectÃ¡ngulo mÃ¡s grande que comienza en (x,y)
                            max_width = 1
                            max_height = 1
                            
                            # Expandir horizontalmente
                            while x + max_width < width and roi[y, x + max_width] == 1:
                                max_width += 1
                            
                            # Expandir verticalmente
                            can_expand = True
                            while y + max_height < height and can_expand:
                                for x2 in range(x, x + max_width):
                                    if x2 >= width or roi[y + max_height, x2] == 0:
                                        can_expand = False
                                        break
                                if can_expand:
                                    max_height += 1
                            
                            # AÃ±adir el rectÃ¡ngulo
                            rectangles.append((min_col + x, min_row + y, max_width, max_height))
                            
                            # Marcar el Ã¡rea como procesada
                            roi[y:y+max_height, x:x+max_width] = 0
        else:
            # Para regiones mÃ¡s grandes, usar un enfoque de divisiÃ³n y conquista
            # Dividir la regiÃ³n en cuadrantes
            half_height = height // 2
            half_width = width // 2
            
            # Procesar cada cuadrante
            quadrants = [
                (0, 0, half_width, half_height),
                (half_width, 0, width - half_width, half_height),
                (0, half_height, half_width, height - half_height),
                (half_width, half_height, width - half_width, height - half_height)
            ]
            
            for qx, qy, qw, qh in quadrants:
                if qw <= 0 or qh <= 0:
                    continue
                    
                # Extraer el cuadrante
                quadrant = roi[qy:qy+qh, qx:qx+qw]
                
                # Si el cuadrante estÃ¡ vacÃ­o, omitirlo
                if np.sum(quadrant) == 0:
                    continue
                
                # Calcular el Ã¡rea cubierta por la mÃ¡scara en este cuadrante
                coverage = np.sum(quadrant) / (qw * qh)
                
                # Si la cobertura es alta, usar un rectÃ¡ngulo que cubra todo el cuadrante
                if coverage > 0.7:
                    rectangles.append((min_col + qx, min_row + qy, qw, qh))
                else:
                    # Procesar recursivamente
                    sub_rectangles = self.decompose_region_to_rectangles(
                        mask, 
                        min_row + qy, 
                        min_col + qx, 
                        min_row + qy + qh, 
                        min_col + qx + qw
                    )
                    rectangles.extend(sub_rectangles)
        
        return rectangles
    
    def fill_with_lego_pieces(self, max_piece_size=8):
        """
        Rellena las formas bÃ¡sicas con piezas LEGO
        optimizando para detalles y forma
        
        Args:
            max_piece_size (int): TamaÃ±o mÃ¡ximo de pieza a utilizar
        
        Returns:
            dict: Piezas LEGO utilizadas por color
        """
        if self.basic_shapes is None:
            raise ValueError("Primero debes transformar las regiones en formas bÃ¡sicas")
        
        self.lego_pieces = defaultdict(list)
        self.total_pieces = 0
        
        for color, shapes in self.basic_shapes.items():
            for shape in shapes:
                if shape['type'] == 'rectangle':
                    # Obtener dimensiones del rectÃ¡ngulo
                    x, y = shape['x'], shape['y']
                    width, height = shape['width'], shape['height']
                    is_detail = shape.get('is_detail', False)
                    
                    # Determinar el tamaÃ±o mÃ¡ximo de pieza para esta Ã¡rea
                    # Usar piezas mÃ¡s pequeÃ±as para Ã¡reas de detalle
                    local_max_size = min(max_piece_size, max(width, height))
                    if is_detail:
                        local_max_size = min(local_max_size, 2)  # Limitar a piezas 1x1, 1x2, 2x1, 2x2 para detalles
                    
                    # Rellenar el rectÃ¡ngulo con piezas LEGO
                    self.fill_rectangle_with_pieces(color, x, y, width, height, local_max_size, is_detail)
        
        return self.lego_pieces
    
    def fill_rectangle_with_pieces(self, color, x, y, width, height, max_piece_size, is_detail=False):
        """
        Rellena un rectÃ¡ngulo con piezas LEGO optimizando para detalles
        
        Args:
            color (tuple): Color RGB
            x, y (int): Coordenadas de la esquina superior izquierda
            width, height (int): Ancho y alto del rectÃ¡ngulo
            max_piece_size (int): TamaÃ±o mÃ¡ximo de pieza a utilizar
            is_detail (bool): Si es una regiÃ³n de detalle
        """
        # Crear una matriz para representar el Ã¡rea a rellenar
        area = np.ones((height, width), dtype=np.int8)
        
        # Mientras queden celdas por rellenar
        while np.sum(area) > 0:
            # Encontrar la primera celda no rellenada
            y_offset, x_offset = np.unravel_index(np.argmax(area), area.shape)
            
            # Si no hay mÃ¡s celdas por rellenar, salir
            if area[y_offset, x_offset] == 0:
                break
            
            # Encontrar la pieza mÃ¡s grande que cabe en esta posiciÃ³n
            best_piece = None
            best_area = 0
            best_width = 0
            best_height = 0
            
            # Para Ã¡reas de detalle, preferir piezas pequeÃ±as
            if is_detail:
                # Probar primero con piezas 1x1 para detalles muy pequeÃ±os
                if width * height <= 2:
                    best_piece = next(p for p in self.lego_bricks if p['width'] == 1 and p['height'] == 1)
                    best_width = 1
                    best_height = 1
                else:
                    # Probar diferentes orientaciones (horizontal y vertical)
                    for orientation in ['horizontal', 'vertical']:
                        for piece in self.lego_bricks:
                            piece_width = piece['width']
                            piece_height = piece['height']
                            
                            # Invertir dimensiones para orientaciÃ³n vertical
                            if orientation == 'vertical' and piece_width != piece_height:
                                piece_width, piece_height = piece_height, piece_width
                            
                            # Limitar el tamaÃ±o mÃ¡ximo de pieza para detalles
                            if piece_width > max_piece_size or piece_height > max_piece_size:
                                continue
                            
                            # Verificar si la pieza cabe en el Ã¡rea restante
                            if (x_offset + piece_width <= width and 
                                y_offset + piece_height <= height):
                                
                                # Verificar si todas las celdas estÃ¡n disponibles
                                can_place = True
                                for i in range(piece_height):
                                    for j in range(piece_width):
                                        if y_offset + i >= height or x_offset + j >= width or area[y_offset + i, x_offset + j] == 0:
                                            can_place = False
                                            break
                                    if not can_place:
                                        break
                                
                                if can_place:
                                    piece_area = piece_width * piece_height
                                    # Para detalles, preferir piezas mÃ¡s pequeÃ±as
                                    if best_piece is None or piece_area < best_area:
                                        best_area = piece_area
                                        best_piece = piece
                                        best_width = piece_width
                                        best_height = piece_height
            else:
                # Para Ã¡reas normales, preferir piezas mÃ¡s grandes
                # Probar diferentes orientaciones (horizontal y vertical)
                for orientation in ['horizontal', 'vertical']:
                    for piece in self.lego_bricks:
                        piece_width = piece['width']
                        piece_height = piece['height']
                        
                        # Invertir dimensiones para orientaciÃ³n vertical
                        if orientation == 'vertical' and piece_width != piece_height:
                            piece_width, piece_height = piece_height, piece_width
                        
                        # Limitar el tamaÃ±o mÃ¡ximo de pieza
                        if piece_width > max_piece_size or piece_height > max_piece_size:
                            continue
                        
                        # Verificar si la pieza cabe en el Ã¡rea restante
                        if (x_offset + piece_width <= width and 
                            y_offset + piece_height <= height):
                            
                            # Verificar si todas las celdas estÃ¡n disponibles
                            can_place = True
                            for i in range(piece_height):
                                for j in range(piece_width):
                                    if y_offset + i >= height or x_offset + j >= width or area[y_offset + i, x_offset + j] == 0:
                                        can_place = False
                                        break
                                if not can_place:
                                    break
                            
                            if can_place:
                                piece_area = piece_width * piece_height
                                # Priorizar piezas que maximizan el Ã¡rea cubierta
                                if piece_area > best_area:
                                    best_area = piece_area
                                    best_piece = piece
                                    best_width = piece_width
                                    best_height = piece_height
            
            # Si no se encontrÃ³ ninguna pieza, usar la mÃ¡s pequeÃ±a (1x1)
            if best_piece is None:
                best_piece = next(p for p in self.lego_bricks if p['width'] == 1 and p['height'] == 1)
                best_width = 1
                best_height = 1
            
            # Marcar el Ã¡rea como rellenada
            for i in range(best_height):
                for j in range(best_width):
                    if y_offset + i < height and x_offset + j < width:
                        area[y_offset + i, x_offset + j] = 0
            
            # Agregar la pieza
            piece_name = best_piece['name']
            piece_id = best_piece['id']
            
            # Si las dimensiones estÃ¡n invertidas, ajustar el nombre
            if best_width != best_piece['width'] and best_height != best_piece['height']:
                # Invertir el nombre para reflejar la orientaciÃ³n
                if 'x' in piece_name:
                    parts = piece_name.split('x')
                    if len(parts) >= 3:  # Formato como "Brick 2x4"
                        piece_name = f"{parts[0]}x{best_height}x{best_width}"
            
            self.lego_pieces[color].append({
                'piece': piece_name,
                'id': piece_id,
                'x': x + x_offset,
                'y': y + y_offset,
                'width': best_width,
                'height': best_height,
                'is_detail': is_detail
            })
            
            self.total_pieces += 1
    
    def optimize_pieces(self, max_piece_size=8):
        """
        Optimiza el uso de piezas LEGO agrupando Ã¡reas contiguas del mismo color
        
        Args:
            max_piece_size (int): TamaÃ±o mÃ¡ximo de pieza a utilizar
            
        Returns:
            dict: Piezas LEGO optimizadas por color
        """
        if self.lego_color_image is None:
            raise ValueError("Primero debes mapear los colores a la paleta LEGO")
        
        # Crear una matriz para representar las piezas
        self.piece_matrix = np.zeros((self.studs_height, self.studs_width, 4), dtype=np.int32)
        
        # Llenar la matriz con los colores de la imagen
        for y in range(self.studs_height):
            for x in range(self.studs_width):
                # Obtener el color como array NumPy
                color_array = self.lego_color_image[y, x]
                # Almacenar RGB en los primeros 3 canales
                self.piece_matrix[y, x, 0:3] = color_array
                # El cuarto canal indica si la celda ya ha sido asignada a una pieza (0=no, 1=sÃ­)
                self.piece_matrix[y, x, 3] = 0
        
        # Diccionario para almacenar las piezas optimizadas
        self.optimized_pieces = defaultdict(list)
        self.total_pieces = 0
        
        # Procesar la matriz para encontrar piezas Ã³ptimas
        for y in range(self.studs_height):
            for x in range(self.studs_width):
                # Si la celda ya estÃ¡ asignada, continuar
                if self.piece_matrix[y, x, 3] == 1:
                    continue
                
                # Obtener el color de la celda actual
                current_color = tuple(self.piece_matrix[y, x, 0:3])
                
                # Encontrar la pieza mÃ¡s grande posible para esta posiciÃ³n
                best_width, best_height = self.find_largest_piece(x, y, current_color, max_piece_size)
                
                # Si se encontrÃ³ una pieza vÃ¡lida
                if best_width > 0 and best_height > 0:
                    # Marcar las celdas como asignadas
                    for i in range(best_height):
                        for j in range(best_width):
                            if y + i < self.studs_height and x + j < self.studs_width:
                                self.piece_matrix[y + i, x + j, 3] = 1
                    
                    # Encontrar la pieza LEGO mÃ¡s adecuada
                    piece_info = self.find_best_lego_piece(best_width, best_height)
                    
                    # Agregar la pieza al diccionario
                    self.optimized_pieces[current_color].append({
                        'piece': piece_info['name'],
                        'id': piece_info['id'],
                        'x': x,
                        'y': y,
                        'width': best_width,
                        'height': best_height,
                        'is_detail': best_width * best_height <= 4  # Considerar detalles si son pequeÃ±os
                    })
                    
                    self.total_pieces += 1
        
        return self.optimized_pieces
    
    def find_largest_piece(self, start_x, start_y, color, max_size):
        """
        Encuentra la pieza mÃ¡s grande posible en una posiciÃ³n dada
        
        Args:
            start_x, start_y (int): Coordenadas de inicio
            color (tuple): Color RGB
            max_size (int): TamaÃ±o mÃ¡ximo de pieza
            
        Returns:
            tuple: (width, height) de la pieza mÃ¡s grande
        """
        # Verificar lÃ­mites
        if start_y >= self.studs_height or start_x >= self.studs_width:
            return 0, 0
        
        # Verificar si la celda ya estÃ¡ asignada
        if self.piece_matrix[start_y, start_x, 3] == 1:
            return 0, 0
        
        # Verificar si el color coincide
        if not np.array_equal(self.piece_matrix[start_y, start_x, 0:3], color):
            return 0, 0
        
        # Encontrar el ancho mÃ¡ximo
        max_width = 1
        while (start_x + max_width < self.studs_width and 
               max_width < max_size and 
               np.array_equal(self.piece_matrix[start_y, start_x + max_width, 0:3], color) and
               self.piece_matrix[start_y, start_x + max_width, 3] == 0):
            max_width += 1
        
        # Encontrar la altura mÃ¡xima
        max_height = 1
        can_expand = True
        while can_expand and max_height < max_size and start_y + max_height < self.studs_height:
            # Verificar toda la fila
            for j in range(max_width):
                if (start_x + j >= self.studs_width or 
                    not np.array_equal(self.piece_matrix[start_y + max_height, start_x + j, 0:3], color) or
                    self.piece_matrix[start_y + max_height, start_x + j, 3] == 1):
                    can_expand = False
                    break
            
            if can_expand:
                max_height += 1
        
        # Verificar si la pieza es vÃ¡lida para LEGO (debe ser rectangular)
        valid_piece = False
        for piece in self.lego_bricks:
            if (piece['width'] == max_width and piece['height'] == max_height) or \
               (piece['width'] == max_height and piece['height'] == max_width):
                valid_piece = True
                break
        
        # Si no es una pieza vÃ¡lida, intentar reducir dimensiones
        if not valid_piece:
            # Intentar reducir ancho
            while max_width > 1:
                max_width -= 1
                for piece in self.lego_bricks:
                    if (piece['width'] == max_width and piece['height'] == max_height) or \
                       (piece['width'] == max_height and piece['height'] == max_width):
                        valid_piece = True
                        break
                if valid_piece:
                    break
            
            # Si aÃºn no es vÃ¡lida, intentar reducir altura
            if not valid_piece:
                max_width = 1  # Resetear ancho
                while max_height > 1:
                    max_height -= 1
                    for piece in self.lego_bricks:
                        if (piece['width'] == max_width and piece['height'] == max_height) or \
                           (piece['width'] == max_height and piece['height'] == max_width):
                            valid_piece = True
                            break
                    if valid_piece:
                        break
            
            # Si aÃºn no es vÃ¡lida, usar 1x1
            if not valid_piece:
                max_width = 1
                max_height = 1
        
        return max_width, max_height
    
    def find_best_lego_piece(self, width, height):
        """
        Encuentra la pieza LEGO mÃ¡s adecuada para las dimensiones dadas
        
        Args:
            width, height (int): Dimensiones requeridas
            
        Returns:
            dict: InformaciÃ³n de la pieza LEGO
        """
        # Buscar pieza exacta
        for piece in self.lego_bricks:
            if (piece['width'] == width and piece['height'] == height) or \
               (piece['width'] == height and piece['height'] == width):
                return piece
        
        # Si no se encuentra, usar la pieza mÃ¡s pequeÃ±a (1x1)
        return next(p for p in self.lego_bricks if p['width'] == 1 and p['height'] == 1)
    
    def calculate_budget(self, avg_piece_cost):
        """
        Calcula el presupuesto basado en el costo promedio por pieza
        
        Args:
            avg_piece_cost (float): Costo promedio por pieza LEGO
        
        Returns:
            float: Presupuesto total
        """
        self.piece_cost = avg_piece_cost
        total_budget = self.total_pieces * avg_piece_cost
        
        return total_budget
    
    def generate_piece_list(self):
        """
        Genera una lista de piezas necesarias
        
        Returns:
            dict: Recuento de piezas por tipo y color
        """
        piece_count = defaultdict(lambda: defaultdict(int))
        
        # Usar piezas optimizadas si estÃ¡n disponibles
        pieces_to_count = self.optimized_pieces if self.optimized_pieces else self.lego_pieces
        
        for color, pieces in pieces_to_count.items():
            color_name = self.get_color_name(color)
            for piece in pieces:
                piece_count[color_name][piece['piece']] += 1
        
        return piece_count
    
    def get_color_name(self, rgb_color):
        """
        Obtiene el nombre del color LEGO mÃ¡s cercano a un color RGB
        
        Args:
            rgb_color (tuple): Color RGB
        
        Returns:
            str: Nombre del color LEGO
        """
        # Primero verificar si este color ya estÃ¡ en nuestro mapeo
        for original_color, (lego_color, lego_name) in self.color_mapping.items():
            if np.array_equal(rgb_color, lego_color):
                return lego_name
        
        # Si no estÃ¡ en el mapeo, calcular la distancia
        min_distance = float('inf')
        closest_color_name = "Unknown"
        
        for name, lego_rgb in self.lego_colors.items():
            distance = np.sqrt(np.sum((np.array(rgb_color) - np.array(lego_rgb)) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                closest_color_name = name
        
        return closest_color_name
    
    def visualize_lego_puzzle(self, use_optimized=True):
        """
        Visualiza el puzzle LEGO con mejor representaciÃ³n visual
        
        Args:
            use_optimized (bool): Si se deben usar las piezas optimizadas
            
        Returns:
            numpy.ndarray: Imagen del puzzle LEGO
        """
        # Crear una imagen en blanco con fondo blanco
        scale_factor = 10  # Factor de escala para hacer la imagen mÃ¡s grande
        img_width = self.studs_width * scale_factor
        img_height = self.studs_height * scale_factor
        lego_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Determinar quÃ© conjunto de piezas usar
        pieces_to_draw = self.optimized_pieces if use_optimized and self.optimized_pieces else self.lego_pieces
        
        # Dibujar cada pieza LEGO
        for color_key, pieces in pieces_to_draw.items():
            # Convertir la clave de tupla a lista para usar como color
            if isinstance(color_key, tuple):
                color_rgb = list(color_key)
            else:
                # Si por alguna razÃ³n no es una tupla, usar un color predeterminado
                color_rgb = [0, 0, 0]
            
            for piece in pieces:
                x, y = piece['x'], piece['y']
                width, height = piece['width'], piece['height']
                is_detail = piece.get('is_detail', False)
                
                # Escalar las coordenadas
                x_scaled = int(x * scale_factor)
                y_scaled = int(y * scale_factor)
                width_scaled = int(width * scale_factor)
                height_scaled = int(height * scale_factor)
                
                # Dibujar el rectÃ¡ngulo de la pieza
                cv2.rectangle(lego_image, 
                             (x_scaled, y_scaled), 
                             (x_scaled + width_scaled, y_scaled + height_scaled), 
                             (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])), 
                             -1)  # -1 para rellenar
                
                # Dibujar el borde de la pieza
                border_color = (0, 0, 0)  # Negro para bordes normales
                border_width = 2
                if is_detail:
                    border_width = 3  # Borde mÃ¡s grueso para detalles
                
                cv2.rectangle(lego_image, 
                             (x_scaled, y_scaled), 
                             (x_scaled + width_scaled, y_scaled + height_scaled), 
                             border_color, 
                             border_width)
                
                # Dibujar los studs
                stud_radius = int(scale_factor * 0.3)
                for i in range(width):
                    for j in range(height):
                        stud_x = x_scaled + int((i + 0.5) * scale_factor)
                        stud_y = y_scaled + int((j + 0.5) * scale_factor)
                        
                        # Dibujar cÃ­rculo para el stud
                        cv2.circle(lego_image, 
                                  (stud_x, stud_y), 
                                  stud_radius, 
                                  (220, 220, 220),  # Color gris claro para los studs
                                  -1)  # Rellenar
                        
                        # Dibujar borde del stud
                        cv2.circle(lego_image, 
                                  (stud_x, stud_y), 
                                  stud_radius, 
                                  (0, 0, 0),  # Borde negro
                                  1)  # Grosor del borde
        
        return lego_image
    
    def visualize_lego_puzzle_direct(self):
        """
        Visualiza el puzzle LEGO directamente desde la imagen con colores LEGO
        para evitar problemas de pÃ©rdida de forma
        
        Returns:
            numpy.ndarray: Imagen del puzzle LEGO
        """
        if self.lego_color_image is None:
            raise ValueError("Primero debes mapear los colores a la paleta LEGO")
        
        # Crear una imagen en blanco con fondo blanco
        scale_factor = 10  # Factor de escala para hacer la imagen mÃ¡s grande
        img_width = self.studs_width * scale_factor
        img_height = self.studs_height * scale_factor
        lego_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Para cada pixel en la imagen con colores LEGO
        for y in range(self.studs_height):
            for x in range(self.studs_width):
                # Obtener el color como array NumPy
                color_array = self.lego_color_image[y, x]
                
                # Convertir a int para OpenCV
                color_int = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
                
                # Escalar las coordenadas
                x_scaled = int(x * scale_factor)
                y_scaled = int(y * scale_factor)
                width_scaled = scale_factor
                height_scaled = scale_factor
                
                # Dibujar el rectÃ¡ngulo de la pieza (1x1)
                cv2.rectangle(lego_image, 
                             (x_scaled, y_scaled), 
                             (x_scaled + width_scaled, y_scaled + height_scaled), 
                             color_int, 
                             -1)  # -1 para rellenar
                
                # Dibujar el borde de la pieza
                cv2.rectangle(lego_image, 
                             (x_scaled, y_scaled), 
                             (x_scaled + width_scaled, y_scaled + height_scaled), 
                             (0, 0, 0), 
                             1)
                
                # Dibujar el stud
                stud_radius = int(scale_factor * 0.3)
                stud_x = x_scaled + int(scale_factor * 0.5)
                stud_y = y_scaled + int(scale_factor * 0.5)
                
                # Dibujar cÃ­rculo para el stud
                cv2.circle(lego_image, 
                          (stud_x, stud_y), 
                          stud_radius, 
                          (220, 220, 220),  # Color gris claro para los studs
                          -1)  # Rellenar
                
                # Dibujar borde del stud
                cv2.circle(lego_image, 
                          (stud_x, stud_y), 
                          stud_radius, 
                          (0, 0, 0),  # Borde negro
                          1)  # Grosor del borde
        
        return lego_image
    
    def save_results(self, output_dir):
        """
        Guarda los resultados del proceso
        
        Args:
            output_dir (str): Directorio de salida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar imÃ¡genes
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.imshow(self.original_image)
        plt.title('Imagen Original')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(self.simplified_image)
        plt.title('Colores Simplificados')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(self.lego_color_image)
        plt.title('Colores LEGO')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        # Usar la visualizaciÃ³n optimizada si estÃ¡ disponible
        if self.optimized_pieces:
            lego_puzzle = self.visualize_lego_puzzle(use_optimized=True)
            plt.title('Puzzle LEGO Optimizado')
        else:
            # Usar la visualizaciÃ³n directa como respaldo
            lego_puzzle = self.visualize_lego_puzzle_direct()
            plt.title('Puzzle LEGO')
        plt.imshow(lego_puzzle)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'proceso_conversion.png'), dpi=300)
        
        # Guardar imagen del puzzle LEGO en alta resoluciÃ³n
        if self.optimized_pieces:
            # Guardar versiÃ³n optimizada
            optimized_puzzle = self.visualize_lego_puzzle(use_optimized=True)
            cv2.imwrite(os.path.join(output_dir, 'puzzle_lego_optimizado.png'), 
                       cv2.cvtColor(optimized_puzzle, cv2.COLOR_RGB2BGR))
            
            # TambiÃ©n guardar la visualizaciÃ³n directa para comparaciÃ³n
            direct_puzzle = self.visualize_lego_puzzle_direct()
            cv2.imwrite(os.path.join(output_dir, 'puzzle_lego_directo.png'), 
                       cv2.cvtColor(direct_puzzle, cv2.COLOR_RGB2BGR))
        else:
            # Si no hay optimizaciÃ³n, guardar solo la visualizaciÃ³n directa
            direct_puzzle = self.visualize_lego_puzzle_direct()
            cv2.imwrite(os.path.join(output_dir, 'puzzle_lego_final.png'), 
                       cv2.cvtColor(direct_puzzle, cv2.COLOR_RGB2BGR))
        
        # Guardar lista de piezas
        piece_list = self.generate_piece_list()
        with open(os.path.join(output_dir, 'lista_piezas.json'), 'w', encoding='utf-8') as f:
            json.dump(piece_list, f, indent=4, ensure_ascii=False)
        
        # Crear informe
        total_budget = self.calculate_budget(self.piece_cost)
        
        with open(os.path.join(output_dir, 'informe.txt'), 'w', encoding='utf-8') as f:
            f.write(f"INFORME DEL PUZZLE LEGO\n")
            f.write(f"=====================\n\n")
            f.write(f"Imagen original: {self.image_path}\n")
            f.write(f"Dimensiones del puzzle: {self.studs_width} x {self.studs_height} studs\n")
            f.write(f"TamaÃ±o fÃ­sico: {self.studs_width * self.STUD_WIDTH / 10:.1f} x {self.studs_height * self.STUD_WIDTH / 10:.1f} cm\n\n")
            
            f.write(f"NÃºmero total de piezas: {self.total_pieces}\n")
            f.write(f"Costo promedio por pieza: {self.piece_cost:.2f} â‚¬\n")
            f.write(f"Presupuesto total: {total_budget:.2f} â‚¬\n\n")
            
            f.write(f"LISTA DE PIEZAS POR COLOR\n")
            f.write(f"========================\n\n")
            
            for color, pieces in piece_list.items():
                f.write(f"Color: {color}\n")
                f.write(f"--------------------\n")
                for piece_name, count in pieces.items():
                    f.write(f"  - {piece_name}: {count} unidades\n")
                f.write("\n")


def main():
    """
    FunciÃ³n principal interactiva
    """
    print("\n=== CONVERSOR DE IMÃGENES A PUZZLES LEGO (VERSIÃ“N OPTIMIZADA) ===\n")
    
    # Solicitar la ruta de la imagen
    image_path = input("Introduce la ruta de la imagen: ")
    
    # Verificar que la imagen existe
    if not os.path.isfile(image_path):
        print(f"Error: No se encontrÃ³ la imagen en la ruta: {image_path}")
        return
    
    # Solicitar la altura deseada en cm
    while True:
        try:
            target_height = float(input("Introduce la altura deseada del puzzle en cm: "))
            if target_height <= 0:
                print("La altura debe ser un nÃºmero positivo.")
                continue
            break
        except ValueError:
            print("Por favor, introduce un nÃºmero vÃ¡lido.")
    
    # Solicitar el costo promedio por pieza
    while True:
        try:
            avg_piece_cost = float(input("Introduce el costo promedio por pieza LEGO (â‚¬): "))
            if avg_piece_cost <= 0:
                print("El costo debe ser un nÃºmero positivo.")
                continue
            break
        except ValueError:
            print("Por favor, introduce un nÃºmero vÃ¡lido.")
    
    # Crear directorio de resultados
    output_dir = "resultados_lego"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nIniciando conversiÃ³n de la imagen {image_path} a un puzzle LEGO...")
    print(f"Altura objetivo: {target_height} cm")
    
    try:
        # Crear el conversor
        converter = LegoPuzzleConverter(image_path, target_height)
        
        # Paso 1: Simplificar colores
        print("Paso 1: Simplificando colores...")
        converter.simplify_colors(n_colors=32)
        
        # Paso 2: Mapear a colores LEGO
        print("Paso 2: Mapeando a colores LEGO...")
        converter.map_to_lego_colors()
        
        # Paso 3: Optimizar piezas LEGO
        print("Paso 3: Optimizando piezas LEGO...")
        converter.optimize_pieces(max_piece_size=8)
        
        # Paso 4: Calcular presupuesto
        print(f"Paso 4: Calculando presupuesto (costo promedio por pieza: {avg_piece_cost}â‚¬)...")
        total_budget = converter.calculate_budget(avg_piece_cost)
        
        # Paso 5: Guardar resultados
        print("Paso 5: Generando resultados...")
        converter.save_results(output_dir)
        
        # Mostrar resumen
        piece_list = converter.generate_piece_list()
        total_pieces = converter.total_pieces
        
        print("\n=== RESUMEN DEL PUZZLE LEGO ===")
        print(f"Imagen original: {image_path}")
        print(f"Dimensiones: {converter.studs_width} x {converter.studs_height} studs")
        print(f"TamaÃ±o fÃ­sico: {converter.studs_width * converter.STUD_WIDTH / 10:.1f} x {converter.studs_height * converter.STUD_WIDTH / 10:.1f} cm")
        print(f"NÃºmero total de piezas: {total_pieces}")
        print(f"Presupuesto total: {total_budget:.2f} â‚¬")
        print(f"Resultados guardados en: {output_dir}")
        print("===============================")
        
    except Exception as e:
        print(f"Error durante la conversiÃ³n: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
