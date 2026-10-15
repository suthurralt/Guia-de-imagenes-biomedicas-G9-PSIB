"""
src/modulo3/dinamica.py — Módulo 3, punto 3.2
"Procesamiento en video y curva temporal fisiológica"

Idea general
------------
1. Se recorre el video frame a frame: ROI -> máscara de la pupila (Módulo 2
   + morfología, ver descriptores.mascara_pupila) -> área, diámetro,
   centroide y forma. Además se guarda el brillo medio de cada frame.
2. Se detectan los ARTEFACTOS (parpadeos, movimientos bruscos, máscaras
   mal segmentadas) y se interpolan los huecos cortos.
3. Se suaviza la curva (mediana + gaussiano) para poder derivarla.
4. Se detecta el ESTÍMULO luminoso como un salto en el brillo del frame:
   la luz que se enciende ilumina toda la cara.
5. Se extraen los parámetros fisiológicos de cada respuesta: latencia,
   amplitud y velocidad de constricción (luz ON) y de redilatación (luz OFF).
6. Validación contra la anotación manual del Módulo 2 (MAE / RMSE).

No se usa aprendizaje automático: todo es segmentación clásica,
morfología, estadística robusta (mediana, MAD) y filtrado de señales 1D.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.modulo1 import carga_muestreo as cm
from src.modulo2 import anotacion as an
from src.modulo2 import segmentacion as seg
from src.modulo2.metricas import dice
from src.modulo3 import descriptores as de

# Piso para el MAD del filtro de Hampel (en px): si la curva es casi
# constante el MAD tiende a 0 y cualquier diferencia mínima sería "outlier".
MAD_MIN_PX = 0.5


# =============================================================================
# 1. RECORRIDO DEL VIDEO
# =============================================================================

def info_video(ruta_video: str | Path) -> dict:
    """fps, cantidad de frames y duración del video."""
    cap = cv2.VideoCapture(str(ruta_video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"fps": fps, "n_frames": n, "duracion_s": n / fps,
                "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def medir_frame(roi_rgb: np.ndarray, cfg: dict, medir_iris: bool = False) -> dict:
    """Segmenta la pupila en una ROI y devuelve sus descriptores de forma.
    Con medir_iris=True también estima el radio del iris (para la escala)."""
    res = de.mascara_pupila(roi_rgb, cfg)
    if res["mascara"] is None:
        return {"detectada": False, "radio_iris_px": np.nan}

    fila = {"detectada": True, "radio_estimado_px": res["pupila"][2]}
    fila.update(de.morfometria(res["mascara"]))
    fila["diametro_px"] = fila.pop("diametro_eq_px")
    fila["radio_iris_px"] = np.nan
    if medir_iris:
        _, iris = seg.estimar_pupila(res["gris"], devolver_iris=True)
        fila["radio_iris_px"] = iris[2]
    return fila


def recorrer_video(ruta_video: str | Path, cfg: dict, roi: dict | None = None,
                   paso_frames: int = 1, paso_iris: int = 10,
                   verbose: bool = True) -> pd.DataFrame:
    """Mide la pupila en cada frame del video.

    roi : ROI en fracciones (si es None, la de config.yaml).
    Devuelve una fila por frame procesado con: frame, t_s, brillo (gris
    medio de TODO el frame, para detectar el estímulo), detectada, área,
    diámetro equivalente, perímetro, centroide, circularidad, excentricidad
    y (cada `paso_iris` frames) el radio del iris.
    """
    roi = cfg["roi"] if roi is None else roi
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    filas, idx = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % paso_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fila = {"frame": idx, "t_s": idx / fps,
                        "brillo": float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())}
                fila.update(medir_frame(cm.recortar_roi(rgb, roi), cfg,
                                        medir_iris=(idx % paso_iris == 0)))
                filas.append(fila)
                if verbose and idx % 100 == 0:
                    print(f"  frame {idx}")
            idx += 1
    finally:
        cap.release()
    curva = pd.DataFrame(filas)
    curva.attrs["fps"] = fps / paso_frames      # frecuencia de muestreo de la curva
    return curva


# =============================================================================
# 2. ESTÍMULO LUMINOSO
# =============================================================================

def detectar_estimulos(t: np.ndarray, brillo: np.ndarray, cfg_est: dict) -> dict:
    """Instantes de encendido y apagado de la luz.

    Cuando la luz se enciende, el brillo medio del frame sube de golpe de un
    frame al siguiente; cuando se apaga, baja. Después la cámara corrige la
    exposición, pero eso es lento (varios frames). Por eso se mira la
    DIFERENCIA de brillo entre frames consecutivos, Δb[n] = b[n] − b[n−1]:

        umbral = max(umbral_salto_min, k_mad · 1,4826 · MAD(Δb))

    El MAD estima el "ruido" normal de Δb sin que los propios saltos lo
    inflen. Saltos separados por menos de `separacion_min_s` se toman como
    un mismo evento. El instante del estímulo es el del PRIMER frame con luz
    (o sin luz): la resolución temporal es de un frame (~33 ms a 30 fps).

    Si en el config hay tiempos manuales (t_encendido_s / t_apagado_s), se
    usan esos.
    """
    db = np.diff(brillo, prepend=brillo[0])
    mad = np.median(np.abs(db - np.median(db)))
    umbral = max(cfg_est["umbral_salto_min"], cfg_est["k_mad"] * 1.4826 * mad)

    def agrupar(indices):
        eventos, ultimo = [], -np.inf
        for i in indices:
            if t[i] - ultimo > cfg_est["separacion_min_s"]:
                eventos.append(float(t[i]))
            ultimo = t[i]
        return eventos

    encendido = cfg_est.get("t_encendido_s")
    apagado = cfg_est.get("t_apagado_s")
    return {
        "encendido": list(encendido) if encendido is not None else agrupar(np.flatnonzero(db > umbral)),
        "apagado": list(apagado) if apagado is not None else agrupar(np.flatnonzero(db < -umbral)),
        "umbral": float(umbral),
        "delta_brillo": db,
    }


# =============================================================================
# 3. ARTEFACTOS: DETECCIÓN, RECHAZO E INTERPOLACIÓN
# =============================================================================

def _rellenar_huecos(x: pd.Series, max_largo: int) -> pd.Series:
    """Interpolación lineal SOLO en huecos (NaN consecutivos) de hasta
    `max_largo` muestras y rodeados de datos válidos. Los huecos más largos
    quedan en NaN: inventar varios cientos de ms de señal sería peor que no
    tenerla."""
    interp = x.interpolate(method="linear", limit_area="inside")
    nan = x.isna().to_numpy()
    salida = x.copy()
    i = 0
    while i < len(nan):
        if nan[i]:
            j = i
            while j < len(nan) and nan[j]:
                j += 1
            if j - i <= max_largo:
                salida.iloc[i:j] = interp.iloc[i:j]
            i = j
        else:
            i += 1
    return salida


def limpiar_senal(curva: pd.DataFrame, cfg_art: dict) -> pd.DataFrame:
    """Marca los frames no confiables y arma la curva limpia.

    Criterios (en este orden; cada frame se queda con el primer motivo):
      1. "sin_pupila"     : estimar_pupila no encontró un iris con contraste
                            (típico de un PARPADEO: el párpado tapa el ojo).
      2. "forma"          : circularidad < circularidad_min o excentricidad >
                            excentricidad_max. La pupila es casi circular; una
                            máscara muy irregular o alargada indica que se
                            "escapó" hacia un reflejo, una pestaña o el párpado
                            (parpadeo parcial).
      3. "salto_centroide": el centroide se movió más de salto_centroide_max_px
                            respecto del frame anterior. Un MOVIMIENTO OCULAR
                            brusco (sacada) desenfoca el frame por movimiento;
                            también salta cuando la segmentación agarra otra
                            estructura.
      4. "outlier"        : filtro de HAMPEL sobre el diámetro. En una ventana
                            móvil de hampel_ventana_s se calcula la mediana y el
                            MAD; si |D − mediana| > k · 1,4826 · MAD el valor es
                            atípico. (1,4826·MAD estima el desvío estándar de
                            datos gaussianos, pero sin que los outliers lo
                            inflen.)
    Los valores rechazados pasan a NaN y los huecos de hasta hueco_max_s se
    interpolan linealmente (columnas *_limpio).
    """
    c = curva.copy()
    fps = curva.attrs.get("fps", 1 / np.median(np.diff(c.t_s)))
    motivo = np.full(len(c), "", dtype=object)

    det = c["detectada"].fillna(False).to_numpy(dtype=bool)
    motivo[~det] = "sin_pupila"

    mala_forma = (c["circularidad"] < cfg_art["circularidad_min"]) | \
                 (c["excentricidad"] > cfg_art["excentricidad_max"])
    motivo[(motivo == "") & mala_forma.to_numpy()] = "forma"

    salto = np.hypot(c["cx"].diff(), c["cy"].diff()).to_numpy()
    motivo[(motivo == "") & (salto > cfg_art["salto_centroide_max_px"])] = "salto_centroide"

    # Hampel solo sobre los que siguen siendo válidos
    d = c["diametro_px"].where(motivo == "")
    ventana = max(3, int(round(cfg_art["hampel_ventana_s"] * fps)) | 1)   # impar
    mediana = d.rolling(ventana, center=True, min_periods=3).median()
    mad = (d - mediana).abs().rolling(ventana, center=True, min_periods=3).median()
    umbral = cfg_art["hampel_k"] * 1.4826 * np.maximum(mad, MAD_MIN_PX)
    outlier = ((d - mediana).abs() > umbral).to_numpy()
    motivo[(motivo == "") & outlier] = "outlier"

    c["motivo"] = motivo
    c["valido"] = motivo == ""
    max_hueco = int(round(cfg_art["hueco_max_s"] * fps))
    c["diametro_limpio_px"] = _rellenar_huecos(c["diametro_px"].where(c.valido), max_hueco)
    c["area_limpia_px"] = _rellenar_huecos(c["area_px"].astype(float).where(c.valido), max_hueco)
    c["interpolado"] = ~c.valido & c.diametro_limpio_px.notna()
    return c


# =============================================================================
# 4. SUAVIZADO
# =============================================================================

def suavizar(x: pd.Series, mediana_frames: int, sigma_frames: float) -> pd.Series:
    """Mediana móvil + filtro gaussiano, respetando los NaN.

    - La MEDIANA (no lineal) elimina picos aislados sin redondear los
      cambios bruscos (igual que el filtro de mediana 2D del Módulo 1).
    - El GAUSSIANO es un pasa-bajos: quita el ruido de alta frecuencia que,
      al derivar, se amplificaría (derivar multiplica cada componente de
      frecuencia f por 2πf).
    Los NaN se tratan con convolución normalizada: se convoluciona la señal
    (con NaN = 0) y, aparte, la máscara de datos válidos; el cociente es el
    promedio ponderado usando solo los datos que existen.
    """
    valido = x.notna().to_numpy()
    m = x.rolling(mediana_frames, center=True, min_periods=1).median().to_numpy(copy=True)
    m[~valido] = np.nan
    if sigma_frames <= 0:
        return pd.Series(m, index=x.index)

    radio = int(np.ceil(3 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-radio, radio + 1) / sigma_frames) ** 2)
    num = np.convolve(np.nan_to_num(m), k, mode="same")
    den = np.convolve(valido.astype(float), k, mode="same")
    salida = np.where(valido & (den > 0), num / np.maximum(den, 1e-12), np.nan)
    return pd.Series(salida, index=x.index)


def agregar_suavizado(curva: pd.DataFrame, cfg_suav: dict) -> pd.DataFrame:
    """Columnas diametro_suave_px, area_suave_px y velocidad_px_s (dD/dt)."""
    c = curva.copy()
    fps = curva.attrs.get("fps", 1 / np.median(np.diff(c.t_s)))
    sigma = cfg_suav["sigma_s"] * fps
    c["diametro_suave_px"] = suavizar(c["diametro_limpio_px"], cfg_suav["mediana_frames"], sigma)
    c["area_suave_px"] = suavizar(c["area_limpia_px"], cfg_suav["mediana_frames"], sigma)
    c["velocidad_px_s"] = np.gradient(c["diametro_suave_px"].to_numpy(), c["t_s"].to_numpy())
    c.attrs = curva.attrs
    return c


# =============================================================================
# 5. PARÁMETROS FISIOLÓGICOS
# =============================================================================

def parametros_respuesta(t: np.ndarray, D: np.ndarray, A: np.ndarray, D_limpio: np.ndarray,
                         interpolado: np.ndarray, t_estimulo: float, t_limite: float,
                         tipo: str, cfg_par: dict) -> dict:
    """Parámetros de UNA respuesta (constricción tras encender la luz o
    redilatación tras apagarla) sobre las curvas suavizadas D(t) y A(t).

    - Basal: mediana de D y de A en los `ventana_basal_s` previos al estímulo.
    - Velocidad: v = dD/dt. Pico de velocidad = máximo de |v| en el sentido
      esperado (negativo en la constricción, positivo en la redilatación)
      dentro de [t_estímulo, t_estímulo + ventana_respuesta_s].
    - Inicio de la respuesta: desde el pico de velocidad se retrocede en el
      tiempo mientras |v| siga por encima de `fraccion_inicio`·|v_pico|. El
      primer instante de ese tramo es el inicio. Se usa la velocidad y no un
      umbral sobre D porque D tiene un sesgo antes del estímulo (ver notebook:
      en oscuridad el reflejo de la ventana tapa parte de la pupila).
    - Latencia = t_inicio − t_estímulo.
    - Extremo: mínimo (constricción) o máximo (redilatación) de D después
      del inicio. Amplitud = |D_extremo − D_basal| (también en área y en %).
    - Velocidad media = (cambio de D entre el inicio y el extremo) / duración.
    Banderas de calidad:
      - "extremo_en_borde": el extremo cae al final de la ventana (la
        respuesta podría seguir: amplitud subestimada).
      - "inicio_en_borde": el inicio coincide con el estímulo (latencia no
        resuelta).
      - "en_interpolado": el inicio o el pico de velocidad caen en un tramo
        interpolado (artefacto): esa velocidad la "inventó" la interpolación.
      - "respuesta_detectada": el cambio desde el inicio supera
        `k_ruido` veces el ruido de la curva limpia en el período basal
        (1,4826·MAD) y no está en un tramo interpolado. Si es False, la
        latencia y las velocidades se dejan en NaN: no hay respuesta medible.
    """
    signo = -1.0 if tipo == "constriccion" else 1.0
    v = np.gradient(D, t)

    basal = (t >= t_estimulo - cfg_par["ventana_basal_s"]) & (t < t_estimulo)
    fin = min(t_estimulo + cfg_par["ventana_respuesta_s"], t_limite)
    ventana = np.flatnonzero((t >= t_estimulo) & (t <= fin))
    salida = {"tipo": tipo, "t_estimulo_s": t_estimulo}
    if ventana.size < 3 or np.all(np.isnan(v[ventana])) or not np.any(basal & ~np.isnan(D)):
        return salida

    d_basal, a_basal = np.nanmedian(D[basal]), np.nanmedian(A[basal])
    x = D_limpio[basal]
    x = x[~np.isnan(x)]
    ruido = 1.4826 * np.median(np.abs(x - np.median(x)))

    vs = signo * v[ventana]                           # > 0 en el sentido esperado
    k_pico = ventana[np.nanargmax(vs)]
    v_pico = signo * v[k_pico]
    k_ini = k_pico
    while k_ini - 1 >= ventana[0] and signo * v[k_ini - 1] >= cfg_par["fraccion_inicio"] * abs(v_pico):
        k_ini -= 1

    tramo = np.arange(k_ini, ventana[-1] + 1)
    elegir = np.nanargmin if tipo == "constriccion" else np.nanargmax
    k_ext = tramo[elegir(D[tramo])]
    a_ext = A[tramo][elegir(A[tramo])]
    duracion = t[k_ext] - t[k_ini]
    en_interpolado = bool(interpolado[k_ini] or interpolado[k_pico])
    detectada = bool(abs(D[k_ext] - D[k_ini]) > cfg_par["k_ruido"] * ruido and not en_interpolado)

    salida.update({
        "t_inicio_s": t[k_ini],
        "latencia_s": t[k_ini] - t_estimulo,
        "t_extremo_s": t[k_ext],
        "D_basal_px": d_basal,
        "D_inicio_px": D[k_ini],
        "D_extremo_px": D[k_ext],
        "amplitud_D_px": abs(D[k_ext] - d_basal),
        "amplitud_D_pct": 100 * abs(D[k_ext] - d_basal) / d_basal,
        "amplitud_desde_inicio_D_px": abs(D[k_ext] - D[k_ini]),
        "A_basal_px2": a_basal,
        "A_extremo_px2": a_ext,
        "amplitud_A_px2": abs(a_ext - a_basal),
        "amplitud_A_pct": 100 * abs(a_ext - a_basal) / a_basal,
        "v_pico_px_s": abs(v_pico),
        "t_v_pico_s": t[k_pico],
        "v_media_px_s": abs(D[k_ext] - D[k_ini]) / duracion if duracion > 0 else np.nan,
        "ruido_basal_px": ruido,
        "respuesta_detectada": detectada,
        "extremo_en_borde": bool(k_ext == ventana[-1]),
        "inicio_en_borde": bool(k_ini == ventana[0]),
        "en_interpolado": en_interpolado,
    })
    if not detectada:
        for col in ["t_inicio_s", "latencia_s", "v_pico_px_s", "t_v_pico_s", "v_media_px_s"]:
            salida[col] = np.nan
    return salida


def extraer_parametros(curva: pd.DataFrame, estimulos: dict, cfg_par: dict) -> pd.DataFrame:
    """Parámetros de todas las respuestas del registro: una constricción por
    cada encendido y una redilatación por cada apagado. La ventana de
    búsqueda de cada respuesta se corta en el estímulo siguiente."""
    t = curva["t_s"].to_numpy()
    D = curva["diametro_suave_px"].to_numpy()
    A = curva["area_suave_px"].to_numpy()
    D_limpio = curva["diametro_limpio_px"].to_numpy()
    interp = curva["interpolado"].to_numpy(dtype=bool)
    eventos = sorted([(te, "constriccion") for te in estimulos["encendido"]] +
                     [(te, "redilatacion") for te in estimulos["apagado"]])
    filas = []
    for n, (te, tipo) in enumerate(eventos):
        t_lim = eventos[n + 1][0] if n + 1 < len(eventos) else t[-1]
        filas.append(parametros_respuesta(t, D, A, D_limpio, interp, te, t_lim, tipo, cfg_par))
    return pd.DataFrame(filas)


def variabilidad(parametros: pd.DataFrame,
                 columnas=("latencia_s", "amplitud_D_px", "amplitud_D_pct",
                           "amplitud_A_px2", "v_pico_px_s")) -> pd.DataFrame:
    """Media, desvío, coeficiente de variación y n de cada parámetro, por
    tipo de respuesta, sobre todos los trials/registros con respuesta
    detectada. Con n = 1 el desvío queda en NaN (no se puede estimar)."""
    cols = [c for c in columnas if c in parametros]
    p = parametros[parametros["respuesta_detectada"].fillna(False).astype(bool)]
    largo = p.melt(id_vars="tipo", value_vars=cols, var_name="parametro").dropna()
    tabla = largo.groupby(["tipo", "parametro"], sort=False)["value"].agg(
        n="count", media="mean", desvio=lambda x: x.std(ddof=1))
    tabla["cv_pct"] = 100 * tabla["desvio"] / tabla["media"].abs()
    return tabla


# =============================================================================
# 6. ESCALA PÍXEL -> MM
# =============================================================================

def mm_por_px(curva: pd.DataFrame, cfg_esc: dict) -> float | None:
    """Factor de conversión de píxeles a milímetros.

    modo "iris": el diámetro horizontal del iris (white-to-white) es casi
    constante entre adultos (≈ 11–12 mm) y está en el mismo plano que la
    pupila, así que sirve de "regla" en cada video:
        mm/px = diametro_iris_mm / (2 · mediana del radio del iris en px)
    Es una aproximación (±10 % entre personas). modo "manual": mm_por_px
    del config (p. ej. medido con una regla en el video). None: solo px.
    """
    modo = cfg_esc.get("modo")
    if modo == "manual":
        return cfg_esc["mm_por_px"]
    if modo == "iris":
        r = curva["radio_iris_px"].median()
        return cfg_esc["diametro_iris_mm"] / (2 * r) if r > 0 else None
    return None


def mm_por_px_regla(frame_rgb: np.ndarray, fila_px: int, x0_px: int, x1_px: int,
                    alto_px: int = 4, periodo_rango_px: tuple = (3, 12)) -> tuple[float, np.ndarray, np.ndarray]:
    """Escala a partir de una regla milimetrada visible en el frame.

    Se toma el perfil de intensidad a lo largo de la regla (promedio de
    `alto_px` filas alrededor de `fila_px`, entre x0_px y x1_px), que es una
    señal casi periódica: una raya oscura por milímetro. Su período se
    obtiene con la Transformada de Fourier (como en el Módulo 1.3): la
    frecuencia con mayor magnitud dentro del rango esperado es 1/período.
    Se aplica una ventana de Hann para reducir la fuga espectral y se
    rellena con ceros (n = 8192) para afinar la lectura de la frecuencia.

    Devuelve (mm/px, frecuencias, magnitud) — las dos últimas para graficar.
    """
    g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    perfil = g[fila_px - alto_px // 2:fila_px + alto_px // 2 + 1, x0_px:x1_px].mean(axis=0)
    perfil = (perfil - perfil.mean()) * np.hanning(perfil.size)
    mag = np.abs(np.fft.rfft(perfil, n=8192))
    f = np.fft.rfftfreq(8192)                    # ciclos por píxel
    rango = (f > 1 / periodo_rango_px[1]) & (f < 1 / periodo_rango_px[0])
    periodo_px = 1 / f[np.argmax(np.where(rango, mag, 0))]   # px por mm
    return 1 / periodo_px, f, mag


# =============================================================================
# 7. PIPELINE COMPLETO DE UN REGISTRO
# =============================================================================

def procesar_registro(ruta_video: str | Path, cfg: dict, roi: dict | None = None,
                      verbose: bool = True) -> dict:
    """Video -> curva temporal + estímulos + parámetros fisiológicos.

    Devuelve {"curva", "estimulos", "parametros", "mm_por_px", "info"}.
    La curva trae columnas en px y, si hay escala, también en mm.
    """
    c3 = cfg["modulo3"]
    info = info_video(ruta_video)
    curva = recorrer_video(ruta_video, cfg, roi, c3["video"]["paso_frames"],
                           c3["video"]["paso_iris"], verbose)
    curva = limpiar_senal(curva, c3["artefactos"])
    curva = agregar_suavizado(curva, c3["suavizado"])
    estimulos = detectar_estimulos(curva.t_s.to_numpy(), curva.brillo.to_numpy(), c3["estimulo"])
    parametros = extraer_parametros(curva, estimulos, c3["parametros"])

    escala = mm_por_px(curva, c3["escala"])
    if escala:
        for col in ["diametro_px", "diametro_limpio_px", "diametro_suave_px"]:
            curva[col.replace("_px", "_mm")] = curva[col] * escala
        curva["velocidad_mm_s"] = curva["velocidad_px_s"] * escala
        for col in ["D_basal_px", "D_extremo_px", "amplitud_D_px", "amplitud_desde_inicio_D_px",
                    "v_pico_px_s", "v_media_px_s"]:
            if col in parametros:
                parametros[col.replace("_px", "_mm")] = parametros[col] * escala
    return {"curva": curva, "estimulos": estimulos, "parametros": parametros,
            "mm_por_px": escala, "info": info}


# =============================================================================
# 8. VALIDACIÓN CONTRA LA ANOTACIÓN MANUAL
# =============================================================================

def ubicar_anotacion(frame_rgb: np.ndarray, imagen_anotada_bgr: np.ndarray) -> tuple[int, int, float]:
    """Dónde estaba la ROI anotada dentro del frame: (x0, y0, correlación).

    Las anotaciones del Módulo 2 se guardaron en coordenadas de la ROI que
    había en ese momento en config.yaml. Si después la ROI cambió, las
    coordenadas no coinciden. Para no depender de eso, se busca la imagen
    anotada (contorno_manual_*.png) dentro del frame con correlación cruzada
    normalizada (cv2.matchTemplate). Se usa solo el canal verde, que no
    cambia si en algún paso se invirtió el orden RGB/BGR.
    """
    res = cv2.matchTemplate(frame_rgb[:, :, 1], imagen_anotada_bgr[:, :, 1], cv2.TM_CCOEFF_NORMED)
    _, maximo, _, (x, y) = cv2.minMaxLoc(res)
    return int(x), int(y), float(maximo)


def _pegar(mascara: np.ndarray, forma: tuple, dx: int, dy: int) -> np.ndarray:
    """Copia `mascara` desplazada (dx, dy) en un lienzo de tamaño `forma`."""
    salida = np.zeros(forma[:2], dtype=np.uint8)
    h, w = mascara.shape
    H, W = forma[:2]
    ys, xs = max(0, dy), max(0, dx)
    ye, xe = min(H, dy + h), min(W, dx + w)
    if ye > ys and xe > xs:
        salida[ys:ye, xs:xe] = mascara[ys - dy:ye - dy, xs - dx:xe - dx]
    return salida


def validar_contra_manual(ruta_video: str | Path, cfg: dict, carpeta_anot: str | Path,
                          curva: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compara el área/diámetro automático con la anotación manual.

    Para cada frame anotado (t en segundos) se calcula:
      - A y D manual (D = diámetro equivalente de la máscara manual),
      - A y D automáticos con el pipeline del Módulo 3 (Watershed + erosión),
      - A y D con Watershed sin morfología (para ver qué aporta la erosión),
      - el valor de la curva suavizada en ese frame (si se pasa `curva`),
      - Dice entre la máscara automática y la manual (alineadas con
        ubicar_anotacion).
    Devuelve (tabla por frame, resumen MAE/RMSE/sesgo, {t: máscaras}).
    """
    carpeta_anot = Path(carpeta_anot)
    anotaciones = dict(sorted(an.cargar_anotaciones(carpeta_anot).items()))
    fps = info_video(ruta_video)["fps"]
    filas, mascaras = [], {}
    for t, m_manual in anotaciones.items():
        frame = cm.extraer_frame(ruta_video, t)
        x0, x1, y0, y1 = cm.roi_a_pixeles(frame.shape, cfg["roi"])
        roi = frame[y0:y1, x0:x1]

        # Ubicar la anotación (si existe la imagen con el contorno)
        ruta_img = carpeta_anot / f"contorno_manual_{t}s.png"
        dx = dy = 0
        corr = np.nan
        if ruta_img.exists():
            ax, ay, corr = ubicar_anotacion(frame, cv2.imread(str(ruta_img)))
            dx, dy = ax - x0, ay - y0
        manual = _pegar(m_manual, roi.shape, dx, dy)

        auto = de.mascara_pupila(roi, cfg)
        ws = seg.segmentar_pupila(roi, cfg)
        fila = {"t_s": t, "frame": int(round(t * fps)), "dx_px": dx, "dy_px": dy,
                "correlacion_ubicacion": corr,
                "A_manual_px2": int((m_manual > 0).sum()),
                "D_manual_px": 2 * np.sqrt((m_manual > 0).sum() / np.pi)}
        for nombre, r in [("auto", auto), ("watershed", ws)]:
            m = r["mascara"]
            a = int((m > 0).sum()) if m is not None else np.nan
            fila[f"A_{nombre}_px2"] = a
            fila[f"D_{nombre}_px"] = 2 * np.sqrt(a / np.pi) if m is not None else np.nan
            fila[f"dice_{nombre}"] = dice(manual, m) if m is not None and corr > 0.5 else np.nan
        if curva is not None:
            k = (curva.frame - fila["frame"]).abs().idxmin()
            fila["D_curva_suave_px"] = curva.loc[k, "diametro_suave_px"]
            fila["A_curva_suave_px2"] = curva.loc[k, "area_suave_px"]
            fila["valido_en_curva"] = bool(curva.loc[k, "valido"])
        filas.append(fila)
        mascaras[t] = {"roi": roi, "manual": manual, "auto": auto["mascara"], "watershed": ws["mascara"]}

    tabla = pd.DataFrame(filas)
    resumen = []
    todos = np.ones(len(tabla), dtype=bool)
    pares = [("watershed", "D_watershed_px", "A_watershed_px2", todos),
             ("auto", "D_auto_px", "A_auto_px2", todos)]
    if curva is not None:
        # El mismo método, pero solo en los frames que la limpieza de artefactos acepta
        pares.append(("auto_frames_validos", "D_auto_px", "A_auto_px2", tabla.valido_en_curva.to_numpy()))
        pares.append(("curva_suave", "D_curva_suave_px", "A_curva_suave_px2", todos))
    for nombre, col_d, col_a, sel in pares:
        ed = (tabla[col_d] - tabla.D_manual_px)[sel]
        ea = (tabla[col_a] - tabla.A_manual_px2)[sel]
        col_dice = "dice_auto" if nombre.startswith("auto") else f"dice_{nombre}"
        resumen.append({"metodo": nombre, "n_frames": int(sel.sum()),
                        "MAE_D_px": ed.abs().mean(), "RMSE_D_px": np.sqrt((ed ** 2).mean()),
                        "sesgo_D_px": ed.mean(),
                        "MAE_A_px2": ea.abs().mean(), "RMSE_A_px2": np.sqrt((ea ** 2).mean()),
                        "sesgo_A_px2": ea.mean(),
                        "dice_medio": tabla[col_dice][sel].mean() if col_dice in tabla else np.nan})
    return tabla, pd.DataFrame(resumen), mascaras


# =============================================================================
# 9. SALIDA ESTANDARIZADA (CSV)
# =============================================================================
# Formato fijo pedido para la entrega final: la curva frame a frame y los
# parámetros fisiológicos. Las columnas en mm solo aparecen si hay escala.

COLUMNAS_CURVA = [
    "frame", "t_s", "brillo", "detectada", "valido", "motivo", "interpolado",
    "area_px", "diametro_px", "area_limpia_px", "diametro_limpio_px",
    "area_suave_px", "diametro_suave_px", "velocidad_px_s",
    "diametro_suave_mm", "velocidad_mm_s",
    "cx", "cy", "perimetro_px", "circularidad", "excentricidad",
]
COLUMNAS_PARAMETROS = [
    "registro", "tipo", "respuesta_detectada", "t_estimulo_s", "t_inicio_s", "latencia_s",
    "t_extremo_s", "D_basal_px", "D_inicio_px", "D_extremo_px", "amplitud_D_px", "amplitud_D_pct",
    "amplitud_desde_inicio_D_px", "A_basal_px2", "A_extremo_px2", "amplitud_A_px2", "amplitud_A_pct",
    "v_pico_px_s", "v_media_px_s", "D_basal_mm", "D_extremo_mm", "amplitud_D_mm",
    "amplitud_desde_inicio_D_mm", "v_pico_mm_s", "v_media_mm_s", "ruido_basal_px",
    "extremo_en_borde", "inicio_en_borde", "en_interpolado",
]


def exportar_csv(resultado: dict, registro: str, carpeta: str | Path) -> tuple[Path, Path]:
    """Guarda <registro>_curva.csv y <registro>_parametros.csv en `carpeta`."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    curva = resultado["curva"]
    par = resultado["parametros"].assign(registro=registro)
    ruta_c = carpeta / f"{registro}_curva.csv"
    ruta_p = carpeta / f"{registro}_parametros.csv"
    curva[[c for c in COLUMNAS_CURVA if c in curva]].to_csv(ruta_c, index=False, float_format="%.4f")
    par[[c for c in COLUMNAS_PARAMETROS if c in par]].to_csv(ruta_p, index=False, float_format="%.4f")
    return ruta_c, ruta_p
