"""
src/modulo2/metricas.py

Métricas de comparación entre una máscara automática y la referencia manual.

- Dice, Jaccard (= IoU), precisión y recall para máscaras rellenas.
- Distancia media y Hausdorff para el esqueleto del contorno, que NO es una
  región rellena y por eso no se puede comparar con Dice/IoU.
"""

import cv2
import numpy as np
import pandas as pd


def _bool(m):
    return np.asarray(m) > 0


def dice(ref, pred):
    """
    Calcula el coeficiente Dice entre la máscara de referencia
    y la predicción. Valores cercanos a 1 indican mayor superposición.
    """
    ref, pred = _bool(ref), _bool(pred)
    suma = ref.sum() + pred.sum()
    if suma == 0:
        return 1.0
    return 2.0 * np.logical_and(ref, pred).sum() / suma


def jaccard(ref, pred):
    """Jaccard = IoU = |A ∩ B| / |A ∪ B|."""
    ref, pred = _bool(ref), _bool(pred)
    union = np.logical_or(ref, pred).sum()
    if union == 0:
        return 1.0
    return np.logical_and(ref, pred).sum() / union


def precision_recall(ref, pred):
    """
    Calcula precisión y recall para la máscara predicha.

    Precisión: proporción de los píxeles predichos como pupila
    que pertenecen a la referencia manual.

    Recall: proporción de los píxeles de la pupila de referencia
    que fueron identificados por el método.
    """
    ref, pred = _bool(ref), _bool(pred)
    inter = np.logical_and(ref, pred).sum()
    precision = inter / pred.sum() if pred.sum() else 0.0
    recall = inter / ref.sum() if ref.sum() else 0.0
    return precision, recall


def _borde(mascara):
    """Extrae el borde de una máscara mediante el gradiente morfológico."""
    m = (_bool(mascara)).astype(np.uint8) * 255
    return cv2.morphologyEx(
        m, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ) > 0


def distancias_contorno(ref, esqueleto):
    """
    Distancia (en píxeles) entre el esqueleto del contorno y el borde de la
    referencia manual. Devuelve (media, hausdorff).
    """
    borde_ref = _borde(ref)
    esq = _bool(esqueleto)
    if not esq.any() or not borde_ref.any():
        return np.nan, np.nan

    # Distancia de cada píxel al borde de referencia más cercano (y viceversa)
    dt_ref = cv2.distanceTransform((~borde_ref).astype(np.uint8), cv2.DIST_L2, 5)
    dt_esq = cv2.distanceTransform((~esq).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = dt_ref[esq]          # esqueleto -> referencia
    d2 = dt_esq[borde_ref]    # referencia -> esqueleto
    media = (d1.mean() + d2.mean()) / 2
    hausdorff = max(d1.max(), d2.max())
    return media, hausdorff


def evaluar_metodos(ref, resultados):
    """
    ref: máscara manual. resultados: {nombre: máscara} (salida de
    morf.comparar_morfologia). Devuelve un dataframe con una fila por método.
    """
    filas = []
    for nombre, pred in resultados.items():
        fila = {"metodo": nombre}
        if "esqueleto" in nombre.lower():
            media, haus = distancias_contorno(ref, pred)
            fila.update(dist_media_px=media, hausdorff_px=haus)
        else:
            p, r = precision_recall(ref, pred)
            fila.update(dice=dice(ref, pred), jaccard=jaccard(ref, pred),
                        precision=p, recall=r)
        filas.append(fila)
    return pd.DataFrame(filas)
