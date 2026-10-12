"""
src/modulo1/calidad_filtrado.py — Módulo 1, punto 1.2
"Carga, Muestreo y Espacios de Color"

Idea general
------------

"""
import cv2
import numpy as np

# 1.2.1 - Histograma y Brillo/Contraste
# Convertir la imagen a escala de grises.
def conv_grises(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Calcular el histograma de la imagen en escala de grises.
def histograma(img_gris): 

    """Cuenta cuántos píxeles hay de cada nivel de gris (0,...,255) y devuelve
    un vector de 256 posiciones"""
    return np.bincount(img_gris.ravel(),minlength=256)

# .ravel() aplana la matriz en una sola fila larga, porque np.bincount() solo acepta listas planas.
# np.bincount cuenta cuántas veces aparece cada número en una lista plana. Si la intensidad 30 aparece 500 veces, la posición 30 del resultado vale 500
