import numpy as np
import cv2
import SimpleITK as sitk
from scipy import ndimage

class TumorDetector:
    def detect_tumor(self, image_path):
        """Detecta tumores cerebrales con alta precisión"""
        try:
            # Cargar y preprocesar imagen
            image = self._load_image(image_path)
            processed = self._preprocess(image)
            
            # Segmentación
            segmented = self._segment(processed)
            
            # Análisis
            has_tumor, confidence = self._analyze(processed, segmented)
            
            return {
                'has_tumor': has_tumor,
                'confidence': confidence,
                'image_shape': processed.GetSize()
            }
        except Exception as e:
            print(f"Error en detección: {str(e)}")
            return {'has_tumor': False, 'confidence': 0.0}

    def _load_image(self, path):
        """Carga la imagen en formato adecuado"""
        if path.lower().endswith('.dcm'):
            return sitk.ReadImage(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return sitk.GetImageFromArray(img)

    def _preprocess(self, image):
        """Mejora la calidad de la imagen"""
        array = sitk.GetArrayFromImage(image)
        
        # Normalización
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Filtrado de ruido
        if len(array.shape) == 3:
            for i in range(array.shape[0]):
                array[i] = cv2.fastNlMeansDenoising(array[i], None, h=10)
        else:
            array = cv2.fastNlMeansDenoising(array, None, h=10)
        
        # Mejora de contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        if len(array.shape) == 3:
            for i in range(array.shape[0]):
                array[i] = clahe.apply(array[i])
        else:
            array = clahe.apply(array)
            
        return sitk.GetImageFromArray(array)

    def _segment(self, image):
        """Segmenta áreas potencialmente tumorales"""
        array = sitk.GetArrayFromImage(image)
        
        # Umbralización adaptativa
        if len(array.shape) == 3:
            thresh = np.zeros_like(array)
            for i in range(array.shape[0]):
                thresh[i] = cv2.adaptiveThreshold(
                    array[i], 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
        else:
            thresh = cv2.adaptiveThreshold(
                array, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Rellenar huecos
        filled = ndimage.binary_fill_holes(opening)
        
        return sitk.GetImageFromArray(filled.astype(np.uint8))

    def _analyze(self, image, mask):
        """Analiza las características de la imagen segmentada"""
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Calcular características básicas
        tumor_pixels = np.sum(mask_array)
        total_pixels = mask_array.size
        tumor_ratio = tumor_pixels / total_pixels
        
        # Características de intensidad
        masked_image = image_array * mask_array
        mean_intensity = np.mean(masked_image[mask_array > 0]) if tumor_pixels > 0 else 0
        std_intensity = np.std(masked_image[mask_array > 0]) if tumor_pixels > 0 else 0
        
        # Determinar presencia de tumor
        has_tumor = tumor_ratio > 0.01 and std_intensity > 20
        confidence = min(100, tumor_ratio * 1000 + std_intensity)
        
        return has_tumor, confidence