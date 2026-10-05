"""
src/modulo1/carga_muestreo.py — Módulo 1, punto 1.1
"Carga, Muestreo y Espacios de Color"

Idea general
------------
Una imagen digital es una función discreta F(x, y) = I: para cada posición
(x, y) de una grilla hay un valor de intensidad I. En la computadora eso es
una MATRIZ de NumPy:
    - Imagen en grises: matriz 2D de forma (alto, ancho).
    - Imagen color:     matriz 3D de forma (alto, ancho, canales), donde cada
                        "capa" es un canal (R, G, B).
Cada elemento es un número entero (típicamente uint8: 0..255).

Este archivo SOLO tiene funciones de cálculo (no grafica nada). Los gráficos
se hacen en el notebook. Separar "cálculo" de "visualización" permite
reutilizar estas funciones en el pipeline automático del final, donde no
queremos que aparezcan ventanas.
"""

from __future__ import annotations  # anotaciones "A | B" en Python 3.8/3.9

from pathlib import Path

import cv2           # OpenCV: lectura de imágenes/video y conversiones
import numpy as np   # NumPy: las imágenes SON arrays de NumPy


# =============================================================================
# 1.1.1 — Obtener el frame y definir la ROI
# =============================================================================

def cargar_imagen(ruta: str | Path) -> np.ndarray:
    """Lee una imagen del disco y la devuelve en RGB, 3 canales.

    Detalles importantes que esta función resuelve:
    1. OpenCV lee en orden **BGR** (azul, verde, rojo), no RGB. Es una herencia
       histórica de la librería. Matplotlib, en cambio, espera RGB. Si no se
       convierte, la piel se ve azulada y el iris cambia de color.
    2. Un PNG puede traer un 4º canal **alfa** (transparencia). Tu captura de
       pantalla es así: forma (721, 592, 4). El alfa no tiene información del
       ojo, así que lo descartamos.
    3. Si la imagen es en escala de grises (2D), la pasamos a 3 canales para
       que el resto del código no tenga que contemplar casos especiales.
    """
    ruta = Path(ruta)

    # Verificamos que exista antes de intentar leer: cv2.imread NO lanza
    # error si el archivo no existe, simplemente devuelve None, y el error
    # aparecería más adelante con un mensaje confuso.
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro la imagen: {ruta}")

    # IMREAD_UNCHANGED: lee "tal cual" (respeta alfa y profundidad de 16 bits
    # si la hubiera). El modo por defecto convertiría todo a 8 bits BGR y
    # perderíamos la posibilidad de reportar la profundidad real.
    img = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV no pudo decodificar el archivo: {ruta}")

    # Algunas imágenes (PNG/TIFF científicos) vienen en 16 bits (0..65535).
    # Todo el TP trabaja en 8 bits, así que la llevamos a uint8 quedándonos
    # con los 8 bits más significativos (dividir por 256 = desplazar 8 bits).
    # Las fotos de celular (JPEG) ya son de 8 bits y no pasan por acá.
    if img.dtype == np.uint16:
        print(f"Aviso: {ruta.name} es de 16 bits; se convierte a 8 bits.")
        img = (img >> 8).astype(np.uint8)

    # img.ndim = cantidad de dimensiones del array.
    if img.ndim == 2:
        # Gris (alto, ancho) -> lo repetimos en 3 canales iguales.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # BGRA -> RGB: reordena canales y descarta el alfa en un solo paso.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # BGR -> RGB: solo reordena los canales.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def extraer_frame(ruta_video: str | Path, tiempo_s: float) -> np.ndarray:
    """Extrae del video el frame más cercano al instante `tiempo_s` (segundos)
    y lo devuelve en RGB.

    Un video es una secuencia de imágenes (frames) mostradas a cierta tasa
    (fps = frames por segundo). El frame en el segundo t es el número
    round(t * fps).
    """
    ruta_video = Path(ruta_video)
    if not ruta_video.exists():
        raise FileNotFoundError(f"No encuentro el video: {ruta_video}")

    # VideoCapture abre el video como un "lector" de frames.
    cap = cv2.VideoCapture(str(ruta_video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)                    # frames por segundo
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total de frames

        # Índice del frame pedido, acotado a [0, n_frames-1] para no
        # salirnos del video si piden un tiempo mayor a su duración.
        idx = int(round(tiempo_s * fps))
        idx = max(0, min(idx, n_frames - 1))

        # Nos posicionamos en ese frame y lo leemos.
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()   # ok=False si no pudo leer
        if not ok:
            raise ValueError(f"No se pudo leer el frame {idx} de {ruta_video}")
    finally:
        # "finally" se ejecuta siempre: liberamos el archivo pase lo que pase.
        cap.release()

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def obtener_frame(cfg: dict, raiz: Path) -> np.ndarray:
    """Decide de dónde sale el frame según el config:
    - Si `entrada.imagen_path` tiene valor -> lee esa imagen.
    - Si es null -> extrae el frame de `entrada.video_path` en `frame_tiempo_s`.
    `raiz` es la raíz del repo, para resolver las rutas relativas del config.
    """
    entrada = cfg["entrada"]
    if entrada.get("imagen_path"):   # .get() devuelve None si la clave falta
        return cargar_imagen(raiz / entrada["imagen_path"])
    return extraer_frame(raiz / entrada["video_path"], entrada["frame_tiempo_s"])


def roi_a_pixeles(forma: tuple, roi: dict | None) -> tuple[int, int, int, int]:
    """Convierte la ROI del config (fracciones 0..1) a índices de píxel.

    Devuelve (x0, x1, y0, y1) en píxeles. Si roi es None, devuelve la
    imagen completa.

    Recordar: en NumPy la imagen se indexa img[fila, columna] = img[y, x].
    Por eso para recortar se usa img[y0:y1, x0:x1] (primero Y, después X).
    """
    alto, ancho = forma[:2]   # shape = (alto, ancho, canales)
    if roi is None:
        return 0, ancho, 0, alto

    # Pasamos de fracción a píxel multiplicando por el tamaño y redondeando.
    x0 = int(round(roi["x0"] * ancho))
    x1 = int(round(roi["x1"] * ancho))
    y0 = int(round(roi["y0"] * alto))
    y1 = int(round(roi["y1"] * alto))

    # Acotamos a los bordes de la imagen (por si alguien pone 1.2 o -0.1)
    # y verificamos que la ROI no quede vacía o invertida.
    x0, x1 = max(0, x0), min(ancho, x1)
    y0, y1 = max(0, y0), min(alto, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI inválida en píxeles: x[{x0},{x1}] y[{y0},{y1}]")
    return x0, x1, y0, y1


def recortar_roi(img: np.ndarray, roi: dict | None) -> np.ndarray:
    """Recorta la ROI. Devuelve una COPIA (.copy()) para que modificar la ROI
    no modifique la imagen original (el slicing de NumPy devuelve una "vista"
    que comparte memoria con el array original)."""
    x0, x1, y0, y1 = roi_a_pixeles(img.shape, roi)
    return img[y0:y1, x0:x1].copy()


# =============================================================================
# 1.1.2 — Dimensiones y profundidad
# =============================================================================

def info_imagen(img: np.ndarray) -> dict:
    """Devuelve ancho, alto, canales, profundidad de bits y rango de valores.

    - img.shape  -> (alto, ancho) o (alto, ancho, canales). OJO: primero alto.
    - img.dtype  -> tipo de dato de cada píxel (uint8, uint16, float32...).
    - itemsize   -> bytes que ocupa cada valor; ×8 = bits por canal.
      uint8 -> 1 byte = 8 bits -> 2^8 = 256 niveles posibles (0..255).
    - Bits por píxel = bits por canal × canales (RGB 8 bits = 24 bpp).
    """
    alto, ancho = img.shape[:2]
    canales = 1 if img.ndim == 2 else img.shape[2]
    bits_canal = img.dtype.itemsize * 8
    return {
        "ancho": ancho,
        "alto": alto,
        "canales": canales,
        "dtype": str(img.dtype),
        "bits_por_canal": bits_canal,
        "bits_por_pixel": bits_canal * canales,
        "niveles_posibles": 2 ** bits_canal,
        # Valores realmente presentes (no siempre se usa todo el rango).
        "min": int(img.min()),
        "max": int(img.max()),
        "total_pixeles": ancho * alto,
    }


# =============================================================================
# 1.1.3 — Resolución espacial (submuestreo)
# =============================================================================

def submuestrear(img: np.ndarray, escala: float,
                 volver_a_tamano_original: bool = True) -> np.ndarray:
    """Reduce la resolución espacial al `escala` (0.5 = 50 %) del original.

    Paso 1 — Reducir: cv2.resize con INTER_NEAREST toma directamente uno de
    cada N píxeles (decimación pura), que es la definición de muestrear con
    menor frecuencia espacial. No promedia: así el efecto de la pérdida de
    muestras es "puro" (y aparece aliasing en bordes finos como pestañas).
    (INTER_AREA promediaría vecinos = filtrar antes de submuestrear, lo cual
    suaviza; es lo correcto para achicar fotos, pero esconde el fenómeno.)

    Paso 2 — Volver a ampliar (opcional): para COMPARAR lado a lado con el
    original hay que mostrarla al mismo tamaño. Se amplía otra vez con
    INTER_NEAREST, que replica cada píxel en un bloque -> se ve la
    "pixelación". La información perdida no vuelve: solo se agranda.
    """
    if not 0 < escala <= 1:
        raise ValueError("La escala debe estar en (0, 1].")

    alto, ancho = img.shape[:2]
    # max(1, ...) evita tamaño 0 si la escala es muy chica (ej. 5 % de 15 px).
    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    # OJO: cv2.resize recibe (ancho, alto) — al revés que shape (alto, ancho).
    chica = cv2.resize(img, (nuevo_ancho, nuevo_alto),
                       interpolation=cv2.INTER_NEAREST)
    if not volver_a_tamano_original:
        return chica
    return cv2.resize(chica, (ancho, alto), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# 1.1.4 — Cuantización (profundidad de bits)
# =============================================================================

def cuantizar(img: np.ndarray, bits: int) -> np.ndarray:
    """Recuantiza una imagen uint8 (8 bits) a `bits` bits (manteniendo uint8
    para poder mostrarla).

    Con k bits hay L = 2^k niveles. Partimos el rango 0..255 en L intervalos
    de ancho  paso = 256 / L  y a cada píxel le asignamos UN valor
    representativo del intervalo al que cae:

        índice = floor(valor / paso)          -> 0 .. L-1
        nuevo  = índice * paso + paso / 2      -> centro del intervalo

    Ejemplo con 2 bits (L=4, paso=64):
        0..63 -> 32 ; 64..127 -> 96 ; 128..191 -> 160 ; 192..255 -> 224
    Usar el centro (y no el borde inferior) minimiza el error medio y evita
    que la imagen se oscurezca sistemáticamente.

    Se aplica a cada canal por separado: en RGB, con 2 bits/canal solo
    quedan 4^3 = 64 colores posibles -> aparecen "bandas" (falso contorno).
    """
    if img.dtype != np.uint8:
        raise TypeError("cuantizar() espera una imagen uint8 (8 bits).")
    if not 1 <= bits <= 8:
        raise ValueError("bits debe estar entre 1 y 8.")

    niveles = 2 ** bits
    paso = 256 // niveles   # división entera: 16 para 4 bits, 64 para 2 bits

    # Pasamos a int32 para que la cuenta no "desborde" (uint8 da la vuelta
    # al pasar 255: 250 + 10 = 4 en uint8). Luego volvemos a uint8.
    indice = img.astype(np.int32) // paso
    nueva = indice * paso + paso // 2
    return np.clip(nueva, 0, 255).astype(np.uint8)


def contar_niveles(img: np.ndarray) -> int:
    """Cantidad de niveles (gris) o colores (RGB) distintos presentes.

    Sirve para verificar la cuantización:
      - gris 4 bits -> como máximo 2^4 = 16 niveles.
      - RGB 2 bits/canal -> como máximo 4·4·4 = 64 colores.
    En color, cada píxel es una terna (R, G, B): se "aplana" la imagen a una
    lista de ternas (reshape a N×3) y se cuentan las ternas únicas (axis=0).
    """
    if img.ndim == 2:
        return int(np.unique(img).size)
    return int(np.unique(img.reshape(-1, img.shape[2]), axis=0).shape[0])


# =============================================================================
# 1.1.5 — Espacios de color
# =============================================================================

def separar_rgb(img_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Separa los canales R, G y B. Cada uno es una matriz 2D que se puede
    ver en escala de grises: blanco = mucho de ese color, negro = nada.

    img[:, :, 0] significa "todas las filas, todas las columnas, canal 0".
    """
    return {"R": img_rgb[:, :, 0], "G": img_rgb[:, :, 1], "B": img_rgb[:, :, 2]}


def rgb_a_hsv(img_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Convierte RGB -> HSV y devuelve los tres componentes.

    HSV separa el COLOR de la LUMINOSIDAD:
      H (Matiz/Hue)        : qué color es (ángulo en la rueda de colores).
      S (Saturación)       : qué tan "puro" es vs. grisáceo.
      V (Valor)            : brillo = max(R, G, B).
    Útil para el TP: la pupila es oscura (V bajo) sin importar el color del
    iris, así que V suele separarla mejor que cualquier canal RGB.

    ¡Rangos de OpenCV para uint8! (no son los "de libro"):
      H: 0..179  (el ángulo 0..360° se divide por 2 para que entre en 8 bits)
      S: 0..255
      V: 0..255
    """
    # COLOR_RGB2HSV porque nuestra imagen ya está en RGB (si estuviera en BGR
    # recién leída por OpenCV habría que usar COLOR_BGR2HSV).
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return {"H": hsv[:, :, 0], "S": hsv[:, :, 1], "V": hsv[:, :, 2]}
