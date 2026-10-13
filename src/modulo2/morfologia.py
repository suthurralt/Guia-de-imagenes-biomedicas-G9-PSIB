"""
src/modulo2/morfologia.py — Morfología matemática sobre la máscara de la pupila.

Paso 1: las cuatro operaciones básicas.
    Erosión      encoge la máscara (borra salientes finos y píxeles sueltos)
    Dilatación   agranda la máscara (tapa huecos chicos)
    Apertura     erosión -> dilatación: quita salientes y manchas chicas
    Cierre       dilatación -> erosión: rellena muescas y huecos (p. ej. el reflejo)

El tamaño del elemento estructurante se da como fracción del radio de la
pupila (factor), no en píxeles, para que sirva con otras resoluciones.
"""
import cv2
import numpy as np


def _u8(m):
    """Cualquier máscara (bool, 0/1, 0/255) -> uint8 con 0 y 255."""
    return (np.asarray(m) > 0).astype(np.uint8) * 255


def elemento(r, factor):
    """Elemento estructurante elíptico de radio ~ factor * r (diámetro impar)."""
    rad = max(1, int(round(factor * r)))
    d = 2 * rad + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def erosion(mascara, ee):
    return cv2.erode(_u8(mascara), ee)


def dilatacion(mascara, ee):
    return cv2.dilate(_u8(mascara), ee)


def apertura(mascara, ee):
    return cv2.morphologyEx(_u8(mascara), cv2.MORPH_OPEN, ee)


def cierre(mascara, ee):
    return cv2.morphologyEx(_u8(mascara), cv2.MORPH_CLOSE, ee)