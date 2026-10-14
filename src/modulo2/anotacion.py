"""
src/modulo2/anotacion.py

Anotación manual del contorno pupilar para validar la
segmentación y la morfología (consigna 2.2.3) y, más adelante, el
diámetro/área automático del Módulo 3.2 (MAE/RMSE).

Idea: se hace clic sobre el borde de la pupila en unos pocos frames; los
puntos se convierten en una máscara binaria de referencia que se guarda en
disco. Una vez guardadas, NO hace falta volver a anotar: el notebook solo las
carga.

Uso (desde un script o terminal, NO desde el backend inline del notebook):
    controles: clic izq = agregar punto | clic der o 'z' = deshacer
               Enter = terminar | Esc = saltear este frame
"""

import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. ANOTACIÓN INTERACTIVA (ventana de OpenCV)
# ============================================================

def anotar_contorno(roi_bgr, titulo="Anotar pupila", escala=2):
    """
    Abre una ventana para marcar puntos sobre el borde de la pupila.

    Parámetros:
        roi_bgr: ROI en BGR (como la devuelve cv2/cm.recortar_roi).
        escala: factor de zoom de la ventana (solo visual; los puntos
                devueltos están en coordenadas de la ROI original).

    Devuelve:
        Lista de (x, y) en coordenadas de la ROI, o None si se salteó.
    """
    base = cv2.resize(
        roi_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC
    )
    puntos = []  # En coordenadas de la ventana (ampliadas)

    def dibujar():
        vista = base.copy()
        for p in puntos:
            cv2.circle(vista, p, 3, (0, 255, 0), -1)
        if len(puntos) > 1:
            cv2.polylines(
                vista, [np.array(puntos, np.int32)], len(puntos) > 2,
                (0, 255, 0), 1
            )
        cv2.imshow(titulo, vista)

    def on_mouse(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            dibujar()
        elif evento == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()
            dibujar()

    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(titulo, on_mouse)
    dibujar()

    resultado = None
    while True:
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (13, 10):            # Aprieto Enter
            if len(puntos) >= 3:
                resultado = [(x / escala, y / escala) for x, y in puntos]
                break
            print("Marcá al menos 3 puntos (o Esc para saltear).")
        elif tecla == 27:                # Aprieto Esc
            break
        elif tecla == ord("z") and puntos: # Aprieto z
            puntos.pop()
            dibujar()

    cv2.destroyWindow(titulo)
    cv2.waitKey(1)
    return resultado


# ============================================================
# 2. PUNTOS -> MÁSCARA
# ============================================================

def puntos_a_mascara(puntos, forma, modo="poligono"):
    """
    Convierte los puntos clickeados en una máscara uint8 (255 = pupila).

    modo:
        'poligono': rellena el polígono que une los puntos.
        'elipse'  : ajusta una elipse a los puntos (necesita >= 5).
                    Suele dar un borde más suave, porque la pupila es
                    casi elíptica.
    """
    pts = np.round(np.array(puntos)).astype(np.int32)
    mascara = np.zeros(forma[:2], dtype=np.uint8)

    if modo == "elipse":
        if len(pts) < 5:
            raise ValueError("El ajuste de elipse necesita al menos 5 puntos.")
        elipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
        cv2.ellipse(mascara, elipse, 255, -1)
    elif modo == "poligono":
        cv2.fillPoly(mascara, [pts], 255)
    else:
        raise ValueError("modo debe ser 'poligono' o 'elipse'.")

    return mascara


# ============================================================
# 3. GUARDAR / CARGAR
# ============================================================

def guardar_anotacion(carpeta, t, puntos, mascara, roi_bgr):
    """Guarda máscara (png), puntos (json) y una imagen con el contorno."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(carpeta / f"mascara_manual_{t}s.png"), mascara)

    with open(carpeta / f"puntos_{t}s.json", "w") as f:
        json.dump([list(map(float, p)) for p in puntos], f)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    superpuesta = roi_bgr.copy()
    cv2.drawContours(superpuesta, contornos, -1, (0, 255, 0), 1)
    cv2.imwrite(str(carpeta / f"contorno_manual_{t}s.png"), superpuesta)


def cargar_anotaciones(carpeta):
    """
    Carga todas las máscaras manuales guardadas.

    Devuelve: {t (float/int, en segundos): máscara uint8 0/255}
    """
    carpeta = Path(carpeta)
    anotaciones = {}
    for ruta in sorted(carpeta.glob("mascara_manual_*s.png")):
        t_txt = ruta.stem.replace("mascara_manual_", "").rstrip("s")
        t = float(t_txt)
        t = int(t) if t.is_integer() else t
        m = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        anotaciones[t] = np.where(m > 0, 255, 0).astype(np.uint8)
    return anotaciones


# ============================================================
# 4. FLUJO COMPLETO
# ============================================================

def anotar_frames(obtener_roi, tiempos, carpeta, modo="poligono",
                  escala=2, rehacer=False):
    """
    Anota una lista de tiempos (s). Saltea los ya anotados salvo rehacer=True.

    obtener_roi: función t -> ROI en BGR, ej.
        lambda t: cm.recortar_roi(cm.extraer_frame(ruta_video, t), cfg["roi"])
    """
    carpeta = Path(carpeta)
    for t in tiempos:
        if (carpeta / f"mascara_manual_{t}s.png").exists() and not rehacer:
            print(f"t = {t} s: ya anotado, se saltea.")
            continue

        roi = obtener_roi(t)
        puntos = anotar_contorno(roi, f"t = {t} s", escala=escala)
        if puntos is None:
            print(f"t = {t} s: salteado.")
            continue

        mascara = puntos_a_mascara(puntos, roi.shape, modo=modo)
        guardar_anotacion(carpeta, t, puntos, mascara, roi)
        print(f"t = {t} s: guardado ({len(puntos)} puntos).")
