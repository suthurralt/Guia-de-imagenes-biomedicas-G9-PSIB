"""
src/modulo3/morfometria.py
 
Descriptores de forma de la pupila a partir de la máscara de Watershed
(255 = pupila, 0 = fondo): área, perímetro, circularidad, excentricidad
y coordenadas del centroide.
"""
import cv2
import numpy as np
 
 
def descriptores_forma(mascara):
    """
    Recibe la máscara binaria (255 = pupila) y devuelve un diccionario con:
 
        area_px        : cantidad de píxeles de la zona blanca (px²)
        perimetro_px   : largo del contorno de la zona blanca (px)
        circularidad   : 4*pi*área / perímetro²   (1 = círculo perfecto)
        excentricidad  : sqrt(1 - (b/a)²)          (0 = círculo, cerca de 1 = alargada)
        xc, yc         : centroide (centro de masa) en píxeles
 
    Devuelve None si la máscara es None o está vacía.
    """
    if mascara is None:
        return None
 
    m = (np.asarray(mascara) > 0).astype(np.uint8)
 
    # Contorno externo más grande (ignora manchas sueltas)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
 
    # Área y perímetro
    area_px = int(m.sum())
    area_contorno = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
 
    # Centroide
    mom = cv2.moments(m, binaryImage=True)
    xc = mom["m10"] / mom["m00"]
    yc = mom["m01"] / mom["m00"]
 
    # Circularidad
    circularidad = 4 * np.pi * area_contorno / perimetro**2 if perimetro > 0 else np.nan
 
    # Excentricidad (momentos centrales de 2º orden)
    mu20 = mom["mu20"] / mom["m00"]
    mu02 = mom["mu02"] / mom["m00"]
    mu11 = mom["mu11"] / mom["m00"]
    raiz = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
    lam1 = (mu20 + mu02 + raiz) / 2          # autovalor mayor
    lam2 = (mu20 + mu02 - raiz) / 2          # autovalor menor
    excentricidad = np.sqrt(1 - lam2 / lam1) if lam1 > 0 else np.nan
 
    return {
        "area_px": area_px,
        "perimetro_px": perimetro,
        "circularidad": circularidad,
        "excentricidad": excentricidad,
        "xc": xc,
        "yc": yc,
    }
from src.modulo2.segmentacion import segmentar_pupila
import src.modulo3.morfometria as mor
