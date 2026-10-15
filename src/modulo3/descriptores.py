"""
src/modulo3/descriptores.py — Módulo 3, punto 3.1
"Descriptores morfométricos y de textura en el frame segmentado"

Idea general
------------
Partimos de la máscara de la pupila que da el pipeline del Módulo 2
(seg.segmentar_pupila: CLAHE + Watershed) y le aplicamos la operación
morfológica elegida en 2.2 (erosión). Sobre esa máscara se calculan:

    1. Descriptores de FORMA: área, perímetro, centroide, circularidad y
       excentricidad.
    2. Textura de PRIMER ORDEN: media, desvío, entropía y asimetría del
       histograma de cada canal (R, G, B) dentro de la pupila y del iris.
    3. Textura de SEGUNDO ORDEN: matriz de co-ocurrencia (GLCM) y sus
       atributos (contraste, homogeneidad, energía, correlación).

Este archivo SOLO calcula (no grafica). Las funciones de forma
(`mascara_pupila`, `morfometria`) las reutiliza también el punto 3.2 para
procesar el video frame a frame.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.modulo2 import morfologia as morf
from src.modulo2 import segmentacion as seg


# =============================================================================
# 0. MÁSCARA DE LA PUPILA (Módulo 2 + morfología elegida en 2.2)
# =============================================================================

def componente_principal(mascara: np.ndarray) -> np.ndarray:
    """Se queda solo con la región conectada más grande de la máscara.

    Después de erosionar puede quedar algún grupito de píxeles suelto; la
    pupila es UNA región, así que el resto se descarta. Se usa el etiquetado
    de componentes conexas (vecindad-8) de OpenCV.
    Devuelve uint8 (255 = pupila).
    """
    m = (mascara > 0).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:                                   # solo fondo
        return np.zeros_like(m)
    # stats[:, 4] = área de cada componente; la fila 0 es el fondo.
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (etiquetas == mayor).astype(np.uint8) * 255


def mascara_pupila(roi_rgb: np.ndarray, cfg: dict) -> dict:
    """Máscara final de la pupila en una ROI a color.

    1) seg.segmentar_pupila (Módulo 2: grises -> CLAHE -> estimar_pupila ->
       Watershed sobre el gradiente de Sobel).
    2) Operación morfológica de `modulo3.morfologia` (por defecto erosión
       k=3 con 2 iteraciones: la que mejor Dice dio contra la anotación
       manual en 2.2, porque corrige la sobresegmentación de Watershed).
    3) Se conserva la componente conexa más grande.

    Devuelve {"gris", "pupila", "mascara"}; "pupila" y "mascara" son None si
    estimar_pupila no encontró un iris con contraste suficiente (parpadeo).
    """
    res = seg.segmentar_pupila(roi_rgb, cfg)
    if res["mascara"] is None:
        return res

    m = cfg["modulo3"]["morfologia"]
    mascara = res["mascara"]
    if m.get("operacion"):
        operacion = getattr(morf, m["operacion"])     # p. ej. morf.erosion
        mascara = operacion(mascara, m["kernel"], m["iteraciones"], m["kernel_forma"])
    mascara = componente_principal(mascara)

    # Si la morfología borró todo (pupila muy chica), se usa la de Watershed.
    if not mascara.any():
        mascara = componente_principal(res["mascara"])
    return {"gris": res["gris"], "pupila": res["pupila"], "mascara": mascara}


# =============================================================================
# 1. DESCRIPTORES MORFOMÉTRICOS (forma)
# =============================================================================

def morfometria(mascara: np.ndarray) -> dict:
    """Descriptores de forma de la pupila a partir de su máscara binaria.

    - Área A: cantidad de píxeles de la máscara.
    - Perímetro P: largo del contorno externo (cv2.findContours +
      cv2.arcLength). El contorno une los CENTROS de los píxeles del borde,
      con pasos de 1 (horizontal/vertical) o √2 (diagonal).
    - Centroide (Xc, Yc): momentos de orden 1 sobre los de orden 0,
          Xc = m10 / m00 ,  Yc = m01 / m00.
    - Diámetro equivalente: el del círculo de igual área, D = 2·√(A/π).
    - Circularidad: C = 4π·A_c / P², donde A_c es el área encerrada por el
      MISMO contorno (cv2.contourArea). Vale 1 para un círculo perfecto y
      baja cuanto más irregular o alargada es la forma. Se usa A_c y no el
      conteo de píxeles para que área y perímetro se midan sobre la misma
      curva; si no, en objetos chicos (~20 px de diámetro) C da > 1.
    - Excentricidad: a partir de los momentos centrales de orden 2
      (la "matriz de covarianza" de la región):
          λ1,2 = (μ20 + μ02)/2 ± √( ((μ20 − μ02)/2)² + μ11² )
          e = √(1 − λ2/λ1)
      e = 0 en un círculo; tiende a 1 en una elipse muy alargada.
    """
    m = (mascara > 0).astype(np.uint8)
    area = int(m.sum())
    vacio = {"area_px": area, "perimetro_px": np.nan, "cx": np.nan, "cy": np.nan,
             "diametro_eq_px": np.nan, "circularidad": np.nan, "excentricidad": np.nan,
             "eje_mayor_px": np.nan, "eje_menor_px": np.nan, "orientacion_grados": np.nan}
    if area < 5:
        return vacio

    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(c, closed=True)
    area_contorno = cv2.contourArea(c)

    mom = cv2.moments(m, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    # Momentos centrales normalizados por el área = varianzas de la región
    mu20, mu02, mu11 = mom["mu20"] / mom["m00"], mom["mu02"] / mom["m00"], mom["mu11"] / mom["m00"]
    centro = (mu20 + mu02) / 2
    radio = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = centro + radio, max(centro - radio, 0.0)

    return {
        "area_px": area,
        "perimetro_px": float(perimetro),
        "cx": float(cx),
        "cy": float(cy),
        "diametro_eq_px": float(2 * np.sqrt(area / np.pi)),
        "circularidad": float(4 * np.pi * area_contorno / perimetro ** 2) if perimetro > 0 else np.nan,
        "excentricidad": float(np.sqrt(1 - l2 / l1)) if l1 > 0 else np.nan,
        # Ejes de la elipse con los mismos momentos (4·σ en cada dirección)
        "eje_mayor_px": float(4 * np.sqrt(l1)),
        "eje_menor_px": float(4 * np.sqrt(l2)),
        # Ángulo del eje mayor respecto de la horizontal (y hacia abajo)
        "orientacion_grados": float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))),
    }


# =============================================================================
# 2. REGIONES DE TEXTURA: PUPILA E IRIS
# =============================================================================

def region_iris(gris: np.ndarray, mascara_pupila: np.ndarray, iris: tuple,
                factor_iris: float, margen_pupila_px: int, fila_min_rel: float,
                k_reflejo: float) -> np.ndarray:
    """Máscara booleana del estroma del iris (sin pupila, reflejos ni párpado).

    iris             : (cx, cy, r) del iris, de seg.estimar_pupila(..., devolver_iris=True).
    factor_iris      : se toma el disco de radio factor_iris·r (sin el limbo,
                       que mezcla iris y esclerótica).
    margen_pupila_px : se saca la pupila dilatada este margen (el borde
                       pupilar mezcla pupila e iris).
    fila_min_rel     : se descartan las filas por encima de cy + fila_min_rel·r
                       (el párpado superior y las pestañas tapan esa parte).
    k_reflejo        : los reflejos corneales son mucho más claros que el iris.
                       Se descartan los píxeles con gris > mediana + k·MAD
                       (MAD = desvío absoluto mediano: una medida de dispersión
                       que, a diferencia del desvío estándar, casi no se ve
                       afectada por los propios reflejos). Se dilatan 1 px
                       para sacar también su halo.
    """
    alto, ancho = gris.shape
    cx, cy, r = iris
    yy, xx = np.mgrid[:alto, :ancho]
    disco = np.hypot(xx - cx, yy - cy) <= factor_iris * r
    disco &= yy >= cy + fila_min_rel * r

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margen_pupila_px + 1,) * 2)
    pupila_ancha = cv2.dilate((mascara_pupila > 0).astype(np.uint8), kernel) > 0
    region = disco & ~pupila_ancha

    valores = gris[region].astype(np.float64)
    mediana = np.median(valores)
    mad = np.median(np.abs(valores - mediana))
    reflejo = (gris > mediana + k_reflejo * 1.4826 * mad).astype(np.uint8)
    reflejo = cv2.dilate(reflejo, np.ones((3, 3), np.uint8)) > 0
    return region & ~reflejo


# =============================================================================
# 3. CUANTIZACIÓN A L NIVELES
# =============================================================================

def cuantizar_niveles(img: np.ndarray, niveles: int) -> np.ndarray:
    """Lleva una imagen uint8 (0..255) a `niveles` niveles: devuelve el
    ÍNDICE del nivel (0..niveles−1), que es lo que necesita la GLCM.

        índice = floor(valor · niveles / 256)

    Es la misma partición en intervalos iguales que cm.cuantizar del Módulo 1
    (32 niveles = 5 bits, 64 = 6 bits), pero devolviendo el número de
    intervalo en lugar de su valor central.
    """
    return (img.astype(np.uint16) * niveles // 256).astype(np.uint8)


# =============================================================================
# 4. TEXTURA DE PRIMER ORDEN (histograma)
# =============================================================================

def primer_orden(canal: np.ndarray, region: np.ndarray, niveles: int) -> dict:
    """Estadísticos del histograma de `canal` dentro de `region` (booleana).

    Media μ, desvío σ y asimetría (skewness) se calculan sobre los valores
    originales 0..255:
        asimetría = E[(x − μ)³] / σ³
    (> 0: cola hacia valores claros; < 0: cola hacia oscuros).

    La entropía se calcula sobre el histograma CUANTIZADO a `niveles`
    niveles, H = −Σ p_k·log2(p_k), en bits. Su máximo es log2(niveles)
    (5 bits con 32 niveles): mide cuán "repartidos" están los grises.
    """
    x = canal[region].astype(np.float64)
    mu, sigma = x.mean(), x.std()
    asim = ((x - mu) ** 3).mean() / sigma ** 3 if sigma > 0 else 0.0

    q = cuantizar_niveles(canal[region], niveles)
    p = np.bincount(q, minlength=niveles) / q.size
    p = p[p > 0]
    return {"media": mu, "desvio": sigma, "entropia_bits": float(-(p * np.log2(p)).sum()),
            "asimetria": float(asim), "n_pixeles": int(x.size)}


# =============================================================================
# 5. TEXTURA DE SEGUNDO ORDEN: GLCM
# =============================================================================
# La GLCM P(i, j) cuenta cuántas veces un píxel de nivel i tiene, a distancia
# d en la dirección θ, un vecino de nivel j. Mide cómo se ORGANIZAN
# espacialmente los grises (el histograma no: dos imágenes con el mismo
# histograma pueden tener texturas completamente distintas).
# Se implementa a mano para poder restringirla a una región de forma
# arbitraria (el anillo del iris): solo se cuentan pares con AMBOS píxeles
# dentro de la región.

def _desplazamiento(d: int, angulo_grados: float) -> tuple[int, int]:
    """(dy, dx) del vecino a distancia d en la dirección θ (0° = derecha,
    90° = arriba; la fila crece hacia abajo, por eso el signo de dy)."""
    t = np.deg2rad(angulo_grados)
    return int(round(-d * np.sin(t))), int(round(d * np.cos(t)))


def glcm(indices: np.ndarray, region: np.ndarray, niveles: int, distancia: int,
         angulo_grados: float) -> np.ndarray:
    """GLCM simétrica y normalizada (suma 1) para un desplazamiento.

    indices : imagen cuantizada (0..niveles−1), ver cuantizar_niveles.
    region  : booleana; solo cuentan pares con los dos píxeles adentro.
    Simétrica = se cuenta (i, j) y también (j, i): no importa el sentido
    del desplazamiento (θ y θ+180° dan lo mismo).
    """
    dy, dx = _desplazamiento(distancia, angulo_grados)
    alto, ancho = indices.shape
    # Recortes de la imagen "origen" y la "desplazada" que se solapan
    y0, y1 = max(0, -dy), min(alto, alto - dy)
    x0, x1 = max(0, -dx), min(ancho, ancho - dx)
    a = indices[y0:y1, x0:x1]
    b = indices[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    validos = region[y0:y1, x0:x1] & region[y0 + dy:y1 + dy, x0 + dx:x1 + dx]

    i, j = a[validos].astype(np.int64), b[validos].astype(np.int64)
    P = np.bincount(i * niveles + j, minlength=niveles * niveles).reshape(niveles, niveles)
    P = P + P.T                                  # simétrica
    total = P.sum()
    return P / total if total > 0 else P.astype(np.float64)


def atributos_glcm(P: np.ndarray) -> dict:
    """Atributos de Haralick de una GLCM normalizada.

    contraste    = Σ (i−j)²·P(i,j)        variaciones locales fuertes -> alto
    homogeneidad = Σ P(i,j) / (1+(i−j)²)  pares de niveles parecidos -> alto
    energía      = Σ P(i,j)²              (segundo momento angular, ASM)
                                           pocas combinaciones muy repetidas
                                           (textura uniforme) -> alto
    correlación  = Σ (i−μi)(j−μj)·P / (σi·σj)  dependencia lineal entre un
                                           píxel y su vecino (−1..1)
    Nota: scikit-image llama "energy" a √ASM; acá se usa ASM.
    """
    n = P.shape[0]
    i, j = np.mgrid[:n, :n]
    mu_i, mu_j = (i * P).sum(), (j * P).sum()
    s_i = np.sqrt(((i - mu_i) ** 2 * P).sum())
    s_j = np.sqrt(((j - mu_j) ** 2 * P).sum())
    return {
        "contraste": float(((i - j) ** 2 * P).sum()),
        "homogeneidad": float((P / (1 + (i - j) ** 2)).sum()),
        "energia": float((P ** 2).sum()),
        "correlacion": float(((i - mu_i) * (j - mu_j) * P).sum() / (s_i * s_j)) if s_i * s_j > 0 else 1.0,
    }


def glcm_promedio(indices: np.ndarray, region: np.ndarray, niveles: int,
                  distancias: list, angulos: list) -> tuple[np.ndarray, dict]:
    """GLCM promediada sobre varias distancias y ángulos (textura sin
    dirección preferida, invariante a la rotación) y sus atributos.
    Devuelve (P_media, atributos promedio de cada GLCM)."""
    matrices, atributos = [], []
    for d in distancias:
        for a in angulos:
            P = glcm(indices, region, niveles, d, a)   # Llama a glcm para cada combinacion de d y ángulo
            matrices.append(P)
            atributos.append(atributos_glcm(P))
    return np.mean(matrices, axis=0), pd.DataFrame(atributos).mean().to_dict()


# =============================================================================
# 6. TABLAS COMPLETAS DE TEXTURA (R, G, B x regiones)
# =============================================================================

CANALES = {"R": 0, "G": 1, "B": 2}


def tabla_primer_orden(roi_rgb: np.ndarray, regiones: dict, niveles: int) -> pd.DataFrame:
    """Primer orden de cada canal en cada región. regiones = {nombre: bool}."""
    filas = []
    for nombre, reg in regiones.items():
        for canal, k in CANALES.items():
            fila = {"region": nombre, "canal": canal}
            fila.update(primer_orden(roi_rgb[:, :, k], reg, niveles))
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_glcm(roi_rgb: np.ndarray, regiones: dict, lista_niveles: list,           # Recibe la ROI, las regiones de la ROI, los niveles de cuantización de la imagen, las distancias entre píxeles y los ángulos para la GLCM
               distancias: list, angulos: list) -> tuple[pd.DataFrame, dict]:
    """Atributos GLCM por región, canal y cantidad de niveles.
    Devuelve (tabla, {(region, canal, niveles): GLCM promedio})."""
    filas, matrices = [], {}
    for niveles in lista_niveles:       # Para cada nivel de cuantizacion (32 o 64)
        for nombre, reg in regiones.items():    # Para cada region (pupila o iris)
            for canal, k in CANALES.items():    # Para cada canal (R, G o B)
                idx = cuantizar_niveles(roi_rgb[:, :, k], niveles)      # Se reduce la cantidad de tonos de gris de 0 a 255 a 0 a niveles-1 (32 o 64)
                # Vamos a tener una matríz de niveles x niveles (32x32 o 64x64) para cada region, canal y nivel de cuantizacion
                P, attr = glcm_promedio(idx, reg, niveles, distancias, angulos) # Para cada combinación de d y theta, llama a glcm() y les calcula los 4 atributos (contraste, homogeneidad, energia y correlacion) y devuelve la GLCM promedio y los atributos promedio de cada GLCM
                matrices[(nombre, canal, niveles)] = P
                filas.append({"niveles": niveles, "region": nombre, "canal": canal, **attr})
    return pd.DataFrame(filas), matrices
