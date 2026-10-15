"""
src/modulo3/textura.py — Módulo 3, punto 3.1 (análisis de texturas)
 
Textura de la pupila / iris por canal de color (R, G, B):
 
  1. Primer orden (estadística global del histograma):
       media, desvío estándar, entropía de Shannon y asimetría (skewness).
  2. Segundo orden (GLCM, matriz de co-ocurrencia de niveles de gris):
       contraste, homogeneidad, energía y correlación.
  3. Discusión: ranking de canales según qué tan "rica" es su textura
       (`ranking_canales`).
 
Todas las funciones reciben la ROI en RGB (uint8) y una máscara binaria
(255 = región a analizar). Solo se usan los píxeles dentro de la máscara.
 
Si la máscara es None o está vacía (p. ej. parpadeo, sin iris detectable)
las funciones devuelven valores NaN en lugar de romper.
 
Uso típico desde el notebook:
 
    from src.modulo3 import textura as tex
    reg = tex.segmentar_regiones(roi, cfg)          # máscaras de pupila e iris
    t_iris = tex.tabla_textura(roi, reg["mascara_iris"], niveles=32, region="iris")
    rank = tex.ranking_canales(t_iris)
"""
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
 
CANALES = ("R", "G", "B")
 
# Atributos de primer y segundo orden (orden de columnas de la tabla)
ATRIBUTOS_1 = ("media", "desvio", "entropia", "asimetria")
ATRIBUTOS_2 = ("contraste", "homogeneidad", "energia", "correlacion")
 
# Criterios para decidir qué canal tiene "más riqueza textural".
# True  -> cuanto MÁS ALTO, más textura (más variación / desorden).
# False -> cuanto MÁS BAJO, más textura (una GLCM concentrada en la diagonal,
#          homogénea y de alta energía, indica una región lisa).
# La media, la asimetría y la correlación NO entran: describen brillo, forma
# del histograma y linealidad, no cuánta textura hay.
CRITERIOS_RIQUEZA = {
    "desvio": True,
    "entropia": True,
    "contraste": True,
    "homogeneidad": False,
    "energia": False,
}
 
 
# ------------------------------------------------------------------
# Auxiliares
# ------------------------------------------------------------------
def cuantizar(canal, niveles=32):
    """uint8 [0, 255] -> enteros [0, niveles - 1]."""
    return (canal.astype(np.float32) * niveles / 256.0).astype(np.int32)
 
 
def _mascara_valida(mascara):
    """Devuelve la máscara booleana, o None si es None / está vacía."""
    if mascara is None:
        return None
    m = np.asarray(mascara) > 0
    return m if m.any() else None
 
 
def mascara_iris(forma, pupila_iris, mascara_pupila):
    """
    Máscara del iris = círculo del iris (de estimar_pupila con devolver_iris=True)
    menos la pupila.
 
    forma          : (alto, ancho) de la ROI
    pupila_iris    : salida de estimar_pupila(gris, devolver_iris=True)
                     -> ((cx, cy, r), (cx, cy, r)), o None
    mascara_pupila : máscara de Watershed (255 = pupila), o None
 
    Devuelve una máscara uint8 (255 = iris) o None si falta alguna entrada.
    """
    if pupila_iris is None or mascara_pupila is None:
        return None
    _, (ix, iy, ir) = pupila_iris
    yy, xx = np.ogrid[:forma[0], :forma[1]]
    disco = ((xx - ix) ** 2 + (yy - iy) ** 2) <= ir ** 2
    m = (disco & (np.asarray(mascara_pupila) == 0)).astype(np.uint8) * 255
    return m if m.any() else None
 
 
def segmentar_regiones(roi_rgb, cfg):
    """
    Pipeline completo ROI -> máscaras de pupila e iris (usa Watershed del Módulo 2).
 
    Devuelve un diccionario:
        "gris"           : ROI en grises + CLAHE sobre la que se segmentó
        "pupila"         : (cx, cy, r) estimados, o None
        "iris"           : (cx, cy, r) del iris, o None
        "mascara_pupila" : uint8 (255 = pupila), o None
        "mascara_iris"   : uint8 (255 = iris sin pupila), o None
 
    Todo es None si no se detecta iris con contraste suficiente (parpadeo).
    """
    from src.modulo2 import segmentacion as seg   # import local: evita dependencia circular
 
    res = seg.segmentar_pupila(roi_rgb, cfg)
    out = {"gris": res["gris"], "pupila": None, "iris": None,
           "mascara_pupila": None, "mascara_iris": None}
    if res["mascara"] is None:
        return out
    pi = seg.estimar_pupila(res["gris"], devolver_iris=True)
    if pi is None:
        return out
    out["pupila"], out["iris"] = pi
    out["mascara_pupila"] = res["mascara"]
    out["mascara_iris"] = mascara_iris(roi_rgb.shape[:2], pi, res["mascara"])
    return out
 
 
# ------------------------------------------------------------------
# Primer orden
# ------------------------------------------------------------------
def primer_orden(roi_rgb, mascara, niveles=32):
    """
    Media, desvío, entropía y asimetría por canal. Devuelve {canal: dict}.
 
    media, desvío y asimetría se calculan sobre los valores originales (0-255).
    La entropía de Shannon (en bits) se calcula sobre el histograma cuantizado
    a `niveles` niveles: H = -sum(p * log2 p). Máximo posible: log2(niveles).
    La asimetría es el tercer momento estandarizado: E[(x - mu)^3] / sigma^3.
    """
    dentro = _mascara_valida(mascara)
    out = {}
    for i, nombre in enumerate(CANALES):
        if dentro is None:
            out[nombre] = {a: np.nan for a in ATRIBUTOS_1}
            continue
        canal = roi_rgb[..., i]
        v = canal[dentro].astype(np.float64)
        media, desvio = v.mean(), v.std()
        asimetria = ((v - media) ** 3).mean() / desvio ** 3 if desvio > 0 else 0.0
 
        q = cuantizar(canal, niveles)[dentro]
        p = np.bincount(q, minlength=niveles) / q.size
        p = p[p > 0]
        entropia = float(-(p * np.log2(p)).sum())
 
        out[nombre] = {"media": float(media), "desvio": float(desvio),
                       "entropia": entropia, "asimetria": float(asimetria)}
    return out
 
 
# ------------------------------------------------------------------
# Segundo orden (GLCM)
# ------------------------------------------------------------------
def glcm(roi_rgb, mascara, niveles=32, distancia=1):
    """
    GLCM por canal usando solo pares de píxeles dentro de la máscara.
 
    Truco: el fondo se marca con el nivel 0 (reservado, los niveles reales van
    de 1 a `niveles`) y luego se descarta su fila/columna, así ningún par que
    toque el fondo cuenta. La matriz es simétrica y se normaliza a
    probabilidades. Se calcula en 4 ángulos (0°, 45°, 90°, 135°).
 
    Devuelve {canal: matriz (niveles, niveles, 1, 4)} o {} si la máscara es inválida.
    """
    m = _mascara_valida(mascara)
    if m is None:
        return {}
    ys, xs = np.where(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
 
    out = {}
    for i, nombre in enumerate(CANALES):
        q = cuantizar(roi_rgb[..., i], niveles)[sl] + 1          # niveles 1..N
        q = np.where(m[sl], q, 0).astype(np.uint8)               # 0 = fuera
        G = graycomatrix(q, [distancia], angulos, levels=niveles + 1,
                         symmetric=True, normed=False)[1:, 1:, :, :]
        out[nombre] = G / np.maximum(G.sum(axis=(0, 1), keepdims=True), 1)
    return out
 
 
def atributos_glcm(roi_rgb, mascara, niveles=32, distancia=1):
    """Contraste, homogeneidad, energía y correlación por canal (promedio de 4 ángulos)."""
    mats = glcm(roi_rgb, mascara, niveles, distancia)
    out = {}
    for nombre in CANALES:
        G = mats.get(nombre)
        if G is None:
            out[nombre] = {a: np.nan for a in ATRIBUTOS_2}
            continue
        out[nombre] = {
            "contraste": float(graycoprops(G, "contrast").mean()),
            "homogeneidad": float(graycoprops(G, "homogeneity").mean()),
            "energia": float(graycoprops(G, "energy").mean()),
            "correlacion": float(graycoprops(G, "correlation").mean()),
        }
    return out
 
 
# ------------------------------------------------------------------
# Tabla resumen
# ------------------------------------------------------------------
def tabla_textura(roi_rgb, mascara, niveles=32, region=""):
    """DataFrame con una fila por canal: primer orden + GLCM."""
    p1 = primer_orden(roi_rgb, mascara, niveles)
    p2 = atributos_glcm(roi_rgb, mascara, niveles)
    filas = [{"region": region, "canal": c, **p1[c], **p2[c]} for c in CANALES]
    return pd.DataFrame(filas).set_index(["region", "canal"])
 
 
# ------------------------------------------------------------------
# Discusión: ¿qué canal aporta más riqueza textural?
# ------------------------------------------------------------------
def ranking_canales(tabla, criterios=None):
    """
    Ordena los canales por riqueza textural.
 
    Para cada criterio de `criterios` (por defecto CRITERIOS_RIQUEZA) asigna el
    puesto 1, 2 o 3 a cada canal (1 = más textura según ese criterio). El
    puntaje final es el promedio de puestos: MENOR puntaje = canal más rico.
 
    tabla : salida de `tabla_textura` (una sola región).
    Devuelve un DataFrame ordenado de mejor a peor, con los puestos por criterio
    y la columna "puntaje". Devuelve None si la tabla tiene NaN.
    """
    criterios = CRITERIOS_RIQUEZA if criterios is None else criterios
    t = tabla.droplevel("region") if "region" in tabla.index.names else tabla
    if t[list(criterios)].isna().any().any():
        return None
    puestos = pd.DataFrame({
        c: t[c].rank(ascending=not mas_alto_es_mas_textura, method="min")
        for c, mas_alto_es_mas_textura in criterios.items()
    })
    puestos["puntaje"] = puestos.mean(axis=1)
    return puestos.sort_values("puntaje")
 
 
def ganador(tabla, criterios=None):
    """Canal con mejor (menor) puntaje en `ranking_canales`, o None."""
    r = ranking_canales(tabla, criterios)
    return None if r is None else r.index[0]
 
 
def coef_variacion(tabla):
    """
    Desvío / media por canal (variabilidad RELATIVA al brillo del canal).
 
    Sirve de control para la discusión: un canal oscuro (p. ej. B en un iris
    marrón) puede tener el mismo desvío absoluto que otro más brillante y, sin
    embargo, mucha más variación relativa, que en parte es ruido del sensor o
    de la compresión y no textura real del estroma.
    """
    t = tabla.droplevel("region") if "region" in tabla.index.names else tabla
    return (t["desvio"] / t["media"]).rename("coef_variacion")