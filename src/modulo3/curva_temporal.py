"""
src/modulo3/curva_temporal.py — Módulo 3, punto 3.2
"Procesamiento en video y curva temporal fisiológica"

Qué hace (en orden)
-------------------
1. serie_temporal      : segmenta la pupila frame a frame y arma la tabla
                         t, área, diámetro, circularidad, excentricidad, centroide.
2. validar_contra_manual: (y validar_contra_serie, aproximación sin video) compara área/diámetro automáticos con las máscaras
                         manuales (carpeta anotaciones_manuales) -> MAE, RMSE, sesgo.
3. detectar_artefactos : marca parpadeos, saltos de la pupila (movimiento ocular)
                         y valores atípicos.
4. limpiar_serie       : saca los frames marcados, interpola huecos CORTOS y suaviza
                         (Savitzky-Golay). Los huecos largos quedan en NaN (no se inventan).
5. parametros_plr      : latencia, amplitud, velocidad de constricción y de redilatación
                         para un estímulo.
6. resumen_trials      : media, desvío y CV entre estímulos/registros.
7. plr_sintetica       : curva de verdad conocida para verificar el método.

Este módulo es solo cálculo: los gráficos y el orden de las llamadas están en
notebooks/03_2_curva_temporal.ipynb. Las funciones 3-7 trabajan solo con la tabla
(no necesitan el video ni `src`), así que se pueden probar sin correr la segmentación. Las funciones 1-2 reutilizan
el pipeline del proyecto (Watershed del Módulo 2 + descriptores del Módulo 3).

Bloque para config.yaml (único archivo que se edita a mano):

    modulo3_2:
      paso_frames: 1            # 1 = todos los frames (mejor resolución para la latencia)
      estimulos_s: []           # <-- COMPLETAR: instante(s) en que se aplica el estímulo (s)
      ventana_basal_s: 1.0      # tramo previo al estímulo que define el área basal
      ventana_respuesta_s: 3.0  # cuánto después del estímulo se busca el mínimo (a ajustar)
      max_hueco_s: 0.5          # huecos más largos que esto NO se interpolan
      sg_ventana_s: 0.3
      sg_orden: 2

Convención: diámetro equivalente D = 2·sqrt(A/π) (igual que el Módulo 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. SEGMENTACIÓN FRAME A FRAME
# =============================================================================

COLUMNAS_SERIE = ["t_s", "frame", "area_px", "diametro_px", "xc", "yc",
                  "circularidad", "excentricidad", "motivo"]


def diametro_desde_area(area):
    """D = 2·sqrt(A/π). Acepta escalares o arrays."""
    return 2.0 * np.sqrt(np.asarray(area, dtype=float) / np.pi)


def _medir_roi(roi_rgb: np.ndarray, cfg: dict):
    """Segmenta una ROI y devuelve (fila, máscara). Usa el pipeline del proyecto.

    motivo = "ok" | "sin_iris" (estimar_pupila devolvió None: típico de un
    parpadeo con el ojo cerrado) | "mascara_vacia".
    """
    # Imports locales: así el resto del módulo se puede usar sin tener `src`.
    from src.modulo2 import morfologia as morf
    from src.modulo2 import segmentacion as seg
    from src.modulo3.morfometria import descriptores_forma

    fila = dict(area_px=np.nan, diametro_px=np.nan, xc=np.nan, yc=np.nan,
                circularidad=np.nan, excentricidad=np.nan, motivo="sin_iris")
    res = seg.segmentar_pupila(roi_rgb, cfg)
    if res["mascara"] is None:
        return fila, None

    mascara = res["mascara"]
    m = cfg["modulo2"]["morfologia"]
    if m.get("operacion_final"):          # misma regla que modulo4.medir_pupila
        op = getattr(morf, m["operacion_final"])
        mascara = op(mascara, m["kernel"], m["iteraciones"], m["kernel_forma"])

    d = descriptores_forma(mascara)
    if d is None:
        fila["motivo"] = "mascara_vacia"
        return fila, mascara

    fila.update(area_px=d["area_px"], diametro_px=float(diametro_desde_area(d["area_px"])),
                xc=d["xc"], yc=d["yc"], circularidad=d["circularidad"],
                excentricidad=d["excentricidad"], motivo="ok")
    return fila, mascara


def serie_temporal(ruta_video, cfg: dict, paso_frames: int = 1,
                   t_ini: float = 0.0, t_fin: float | None = None) -> pd.DataFrame:
    """Recorre el video y mide la pupila en 1 de cada `paso_frames` frames.

    Devuelve un DataFrame con COLUMNAS_SERIE (o None si el video no existe). La tasa efectiva de muestreo
    queda en df.attrs["fs_hz"] (= fps / paso_frames).
    """
    import cv2
    from src.modulo1 import carga_muestreo as cm

    ruta_video = Path(ruta_video)
    if not ruta_video.exists():
        return None                      # sin video no hay serie (el notebook lo avisa)

    cap = cv2.VideoCapture(str(ruta_video))
    filas = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if t_fin is not None and t > t_fin:
                break
            if idx % paso_frames == 0 and t >= t_ini:
                roi = cm.recortar_roi(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cfg["roi"])
                fila, _ = _medir_roi(roi, cfg)
                filas.append({"t_s": t, "frame": idx, **fila})
            idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(filas, columns=COLUMNAS_SERIE)
    df.attrs["fs_hz"] = fps / paso_frames
    df.attrs["fps_video"] = fps
    return df


# =============================================================================
# 2. VALIDACIÓN CONTRA LA MEDICIÓN MANUAL (MAE / RMSE)
# =============================================================================

def errores(auto, manual) -> dict:
    """MAE, RMSE y sesgo (auto - manual) ignorando pares con NaN.

    Devuelve también el error relativo medio (% del valor manual).
    """
    a = np.asarray(auto, dtype=float)
    m = np.asarray(manual, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(m))
    if ok.sum() == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, sesgo=np.nan, mae_pct=np.nan)
    e = a[ok] - m[ok]
    return dict(n=int(ok.sum()),
                mae=float(np.mean(np.abs(e))),
                rmse=float(np.sqrt(np.mean(e ** 2))),
                sesgo=float(np.mean(e)),
                mae_pct=float(100 * np.mean(np.abs(e) / m[ok])))


def validar_contra_manual(ruta_video, cfg: dict, carpeta_anotaciones, dice_min: float = 0.5):
    """Mide automáticamente los frames anotados a mano y compara.

    carpeta_anotaciones: results/modulo2/anotaciones_manuales (máscaras
    `mascara_manual_<t>s.png`, hechas con src/modulo2/anotacion.py sobre la ROI).

    Devuelve (tabla por frame, resumen MAE/RMSE de área y de diámetro),
    o (None, None) si falta el video o las máscaras.
    El resumen trae DOS subconjuntos, para no esconder nada:
      - "todos": incluye las fallas groseras (una sola puede dominar el RMSE).
      - "dice >= dice_min": solo frames donde la máscara automática se superpone
        razonablemente con la manual (dice_min = 0,5 es un criterio a justificar).
    Un frame donde la segmentación automática devuelve None (NaN) no entra en
    ninguno, pero queda contado en `n_fallas_nan`.
    """
    from src.modulo1 import carga_muestreo as cm
    from src.modulo2.anotacion import cargar_anotaciones
    from src.modulo2.metricas import dice

    manuales = cargar_anotaciones(carpeta_anotaciones)
    if not manuales or not Path(ruta_video).exists():
        return None, None                # sin anotaciones o sin video no se puede validar

    filas = []
    for t, m_man in sorted(manuales.items()):
        frame = cm.extraer_frame(ruta_video, t)
        fila_auto, m_auto = _medir_roi(cm.recortar_roi(frame, cfg["roi"]), cfg)
        a_man = int(np.count_nonzero(m_man))
        filas.append({
            "t_s": t,
            "area_manual_px": a_man,
            "diametro_manual_px": float(diametro_desde_area(a_man)),
            "area_auto_px": fila_auto["area_px"],
            "diametro_auto_px": fila_auto["diametro_px"],
            "dice": dice(m_man, m_auto) if m_auto is not None else np.nan,
            "motivo_auto": fila_auto["motivo"],
        })
    tabla = pd.DataFrame(filas)

    subconjuntos = (("todos", np.ones(len(tabla), dtype=bool)),
                    (f"dice >= {dice_min}", (tabla["dice"] >= dice_min).to_numpy()))
    magnitudes = (("area_px2", "area_auto_px", "area_manual_px"),
                  ("diametro_px", "diametro_auto_px", "diametro_manual_px"))
    filas_res = []
    for nombre, sel in subconjuntos:
        sub = tabla[sel]
        for mag, c_auto, c_man in magnitudes:
            filas_res.append({"subconjunto": nombre, "magnitud": mag,
                              **errores(sub[c_auto], sub[c_man])})
    resumen = pd.DataFrame(filas_res)
    resumen["n_fallas_nan"] = int(tabla["area_auto_px"].isna().sum())
    return tabla, resumen


def validar_contra_serie(manuales: dict, serie: pd.DataFrame, dice_min: float = 0.5):
    """Adelanto SIN video: compara las máscaras manuales con la serie ya medida.

    manuales : {t_s: máscara} (src.modulo2.anotacion.cargar_anotaciones)
    serie    : tabla con t_s, area_px, diametro_px (p. ej. results/modulo4/4_diametro_video.csv)

    Para cada anotación se toma la fila MÁS CERCANA EN EL TIEMPO, así que el frame
    comparado puede estar corrido hasta medio paso de muestreo: es una aproximación,
    no reemplaza a validar_contra_manual. Como acá no hay máscara automática, no hay
    Dice; el criterio de "falla grosera" es D_auto > 1.5·D_manual (a justificar).
    Devuelve (tabla, resumen) o (None, None) si no hay datos.
    """
    if not manuales or serie is None or len(serie) == 0:
        return None, None
    filas = []
    for t, m in sorted(manuales.items()):
        j = (serie["t_s"] - t).abs().idxmin()
        a_man = int(np.count_nonzero(m))
        filas.append({"t_manual_s": t, "t_serie_s": float(serie["t_s"][j]),
                      "desfase_s": float(serie["t_s"][j] - t),
                      "area_manual_px": a_man,
                      "area_auto_px": float(serie["area_px"][j]),
                      "diametro_manual_px": float(diametro_desde_area(a_man)),
                      "diametro_auto_px": float(serie["diametro_px"][j])})
    tabla = pd.DataFrame(filas)
    tabla["falla_grosera"] = tabla["diametro_auto_px"] > 1.5 * tabla["diametro_manual_px"]

    filas_res = []
    for nombre, sel in (("todos", np.ones(len(tabla), dtype=bool)),
                        ("sin fallas groseras", ~tabla["falla_grosera"].to_numpy())):
        sub = tabla[sel]
        for mag, c_auto, c_man in (("area_px2", "area_auto_px", "area_manual_px"),
                                   ("diametro_px", "diametro_auto_px", "diametro_manual_px")):
            filas_res.append({"subconjunto": nombre, "magnitud": mag,
                              **errores(sub[c_auto], sub[c_man])})
    return tabla, pd.DataFrame(filas_res)


# =============================================================================
# 3. ARTEFACTOS: PARPADEOS, MOVIMIENTOS OCULARES, ATÍPICOS
# =============================================================================
# Fuentes de artefactos y cómo se detectan:
#
#  a) Parpadeo completo: el párpado tapa el iris -> estimar_pupila devuelve None
#     (contraste del iris < contraste_min) -> fila con NaN, motivo "sin_iris".
#  b) Parpadeo parcial / párpado o pestañas sobre la pupila: la máscara deja de
#     parecerse a una elipse -> caen la circularidad y sube la excentricidad, y
#     el área da un salto brusco. Se detecta con puntajes robustos (mediana/MAD).
#  c) Movimiento ocular (sacada, deriva): el centroide de la pupila salta respecto
#     de su trayectoria local (mediana móvil). Además la pupila se ve más ovalada
#     de costado, así que el área proyectada baja sin que cambie el tamaño real.
#  d) Atípicos puntuales (reflejos, sombras, error de Watershed): filtro de Hampel
#     sobre log(área): marca puntos que se alejan > k·1,4826·MAD de la mediana móvil.
#
# Como los parpadeos duran varios frames, cada marca se DILATA unos décimos de
# segundo (los frames vecinos al parpadeo suelen estar contaminados).

def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if x.size == 0 else float(np.median(np.abs(x - np.median(x))))


def _z_robusto(s: pd.Series) -> pd.Series:
    """(x - mediana)/(1,4826·MAD). Si la MAD es 0, devuelve 0 (no marca nada)."""
    mad = _mad(s)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.median()) / (1.4826 * mad)


def _ventana_impar(segundos: float, fs: float, minimo: int = 3) -> int:
    n = max(int(round(segundos * fs)), minimo)
    return n if n % 2 == 1 else n + 1


def _quitar_escalones(la: np.ndarray, marcas: np.ndarray, m: int) -> np.ndarray:
    """Desmarca los tramos marcados tras los cuales la señal NO vuelve al nivel previo.

    Para cada tramo consecutivo de marcas [i, j): nivel previo = mediana de las m
    muestras no marcadas anteriores; nivel posterior = idem posteriores. Si el cambio
    de nivel (|post - pre|) supera la mitad de cuánto se aleja el tramo del nivel
    previo, el tramo es un escalón (p. ej. una constricción real) y no un artefacto.
    Si falta alguno de los dos lados (borde del registro) el tramo se deja marcado.
    """
    marcas = marcas.copy()
    base = np.where(marcas, np.nan, la)
    n, i = len(la), 0
    while i < n:
        if marcas[i]:
            j = i
            while j < n and marcas[j]:
                j += 1
            pre = base[max(0, i - m):i]
            post = base[j:j + m]
            pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
            if pre.size and post.size:
                nivel_pre = np.median(pre)
                cambio = abs(np.median(post) - nivel_pre)
                desvio = np.median(np.abs(la[i:j] - nivel_pre))
                if cambio > 0.5 * desvio:
                    marcas[i:j] = False
            i = j
        else:
            i += 1
    return marcas


def _dilatar(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask
    out = mask.copy()
    for k in range(1, n + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def detectar_artefactos(df: pd.DataFrame, fs_hz: float | None = None,
                        k_forma: float = 4.0, k_centro: float = 4.0,
                        k_hampel: float = 3.0, hampel_ventana_s: float = 1.0,
                        piso_mad_log: float = 0.02, dilatar_s: float = 0.1,
                        col_area: str = "area_px") -> pd.DataFrame:
    """Agrega columnas booleanas de artefacto y `artefacto` (OR de todas).

    Las columnas de forma (circularidad, excentricidad) y de posición (xc, yc)
    son opcionales: si no están, ese criterio se omite.
    Parámetros (todos ajustables; no son valores validados para tu video):
        k_forma, k_centro : umbral en "desvíos robustos" (MAD) de cada criterio.
        k_hampel          : idem para el filtro de Hampel sobre log(área).
        piso_mad_log      : MAD mínima de log(área) (evita marcar todo en tramos
                            muy planos); 0,02 ~ 2 % de cambio de área.
    """
    d = df.copy().reset_index(drop=True)
    if fs_hz is None:
        fs_hz = df.attrs.get("fs_hz") or 1.0 / np.median(np.diff(d["t_s"].unique()))
    area = d[col_area].astype(float)

    # (a) sin segmentación
    d["art_sin_seg"] = area.isna().to_numpy()

    # (b) forma
    art_forma = np.zeros(len(d), dtype=bool)
    if "circularidad" in d and d["circularidad"].notna().any():
        art_forma |= (_z_robusto(d["circularidad"]) < -k_forma).to_numpy()
    if "excentricidad" in d and d["excentricidad"].notna().any():
        art_forma |= (_z_robusto(d["excentricidad"]) > k_forma).to_numpy()
    d["art_forma"] = art_forma

    # (c) salto del centroide respecto de su mediana móvil
    art_centro = np.zeros(len(d), dtype=bool)
    if {"xc", "yc"} <= set(d.columns) and d["xc"].notna().any():
        w = _ventana_impar(hampel_ventana_s, fs_hz)
        mx = d["xc"].rolling(w, center=True, min_periods=3).median()
        my = d["yc"].rolling(w, center=True, min_periods=3).median()
        salto = np.hypot(d["xc"] - mx, d["yc"] - my)
        thr = salto.median() + k_centro * 1.4826 * _mad(salto)
        art_centro = (salto > max(thr, 1.0)).fillna(False).to_numpy()   # piso: 1 px
    d["art_centro"] = art_centro

    # (d) Hampel sobre log(área), usando solo los puntos aún no marcados.
    previo = d["art_sin_seg"] | d["art_forma"] | d["art_centro"]
    la = np.log(area.where(~previo))
    w = _ventana_impar(hampel_ventana_s, fs_hz)
    med = la.rolling(w, center=True, min_periods=3).median()
    mad = (la - med).abs().rolling(w, center=True, min_periods=3).median()
    mad = mad.clip(lower=piso_mad_log)
    atip = ((la - med).abs() > k_hampel * 1.4826 * mad).fillna(False).to_numpy()
    # Protección contra constricciones reales: un tramo marcado solo se mantiene si
    # la señal VUELVE al nivel de antes (pico); si queda en otro nivel (escalón),
    # es fisiología y se desmarca.
    atip = _quitar_escalones(la.to_numpy(), atip, m=max(int(round(0.4 * fs_hz)), 2))
    d["art_atipico"] = atip

    bruto = (d["art_sin_seg"] | d["art_forma"] | d["art_centro"] | d["art_atipico"]).to_numpy()
    d["artefacto"] = _dilatar(bruto, int(round(dilatar_s * fs_hz)))
    d.attrs.update(df.attrs)
    d.attrs["fs_hz"] = fs_hz
    return d


# =============================================================================
# 4. LIMPIEZA: INTERPOLAR HUECOS CORTOS Y SUAVIZAR
# =============================================================================

def _interpolar_huecos_cortos(t: np.ndarray, x: np.ndarray, max_muestras: int) -> np.ndarray:
    """Interpola linealmente SOLO los tramos de NaN de hasta `max_muestras`
    que tienen dato válido a ambos lados. Los huecos más largos quedan en NaN."""
    y = x.copy()
    nan = np.isnan(y)
    i, n = 0, len(y)
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_muestras:
                y[i:j] = np.interp(t[i:j], [t[i - 1], t[j]], [y[i - 1], y[j]])
            i = j
        else:
            i += 1
    return y


def limpiar_serie(df: pd.DataFrame, col: str = "area_px", max_hueco_s: float = 0.5,
                  sg_ventana_s: float = 0.3, sg_orden: int = 2) -> pd.DataFrame:
    """Devuelve df con: <col>_limpia (artefactos fuera + huecos cortos interpolados),
    <col>_suave (Savitzky-Golay) y d<col>_dt (derivada en unidades/s).

    El suavizado se hace tramo por tramo (entre huecos que quedaron en NaN), así
    no se mezclan datos de lados distintos de un parpadeo largo.
    """
    from scipy.signal import savgol_filter

    d = df.copy().reset_index(drop=True)
    fs = d.attrs.get("fs_hz") or 1.0 / np.median(np.diff(d["t_s"].unique()))
    t = d["t_s"].to_numpy(dtype=float)
    x = d[col].astype(float).where(~d["artefacto"]).to_numpy()

    limpia = _interpolar_huecos_cortos(t, x, int(round(max_hueco_s * fs)))
    suave = np.full_like(limpia, np.nan)
    deriv = np.full_like(limpia, np.nan)
    w = _ventana_impar(sg_ventana_s, fs, minimo=sg_orden + 2)

    valido = ~np.isnan(limpia)
    i, n = 0, len(limpia)
    while i < n:
        if valido[i]:
            j = i
            while j < n and valido[j]:
                j += 1
            seg = limpia[i:j]
            wl = min(w, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            if wl >= sg_orden + 2:
                suave[i:j] = savgol_filter(seg, wl, sg_orden)
                deriv[i:j] = savgol_filter(seg, wl, sg_orden, deriv=1, delta=1.0 / fs)
            else:                                   # tramo demasiado corto: sin suavizar
                suave[i:j] = seg
            i = j
        else:
            i += 1

    d[f"{col}_limpia"] = limpia
    d[f"{col}_suave"] = suave
    d[f"d{col}_dt"] = deriv
    d.attrs.update(df.attrs)
    return d


# =============================================================================
# 5. PARÁMETROS FISIOLÓGICOS DEL REFLEJO FOTOMOTOR
# =============================================================================

def _cruce_lineal(t0, x0, t1, x1, nivel):
    """Instante en que el segmento (t0,x0)-(t1,x1) cruza `nivel`."""
    if x1 == x0:
        return t1
    return t0 + (nivel - x0) * (t1 - t0) / (x1 - x0)


def parametros_plr(df: pd.DataFrame, t_estimulo: float, col: str = "area_px",
                   ventana_basal_s: float = 1.0, ventana_respuesta_s: float = 3.0,
                   t_siguiente: float | None = None,
                   k_latencia: float = 3.0, frac_latencia: float = 0.03,
                   n_sostenido: int = 3, fraccion_recuperacion: float = 0.75) -> dict:
    """Parámetros de UNA respuesta (un estímulo) a partir de la serie suavizada.

    Definiciones
    ------------
    basal            : mediana de <col>_suave en [t_est - ventana_basal_s, t_est).
                       Debe ser el tramo en oscuridad.
    mínimo           : mínimo de <col>_suave en [t_est, t_est + ventana_respuesta_s]
                       (o hasta t_siguiente si es menor).
    amplitud         : basal - mínimo (en la unidad de <col>); también en % del basal.
    latencia (umbral): primer instante >= t_est en que la curva baja del umbral
                       basal - max(k_latencia·σ_basal, frac_latencia·basal) durante
                       `n_sostenido` muestras seguidas, con interpolación lineal en el
                       cruce, menos t_est. σ_basal = 1,4826·MAD del tramo basal.
    latencia (tang.) : instante en que la tangente en el punto de máxima velocidad de
                       constricción corta el nivel basal, menos t_est. Es menos sensible
                       al umbral; si difieren mucho, mirar el gráfico.
    vel. constricción: máxima (pico de -dA/dt entre inicio y mínimo) y media
                       (amplitud/(t_mínimo - t_inicio)).
    vel. redilatación: máxima (pico de dA/dt después del mínimo) y media entre el
                       mínimo y el instante en que se recupera `fraccion_recuperacion`
                       de la amplitud (NaN si no se llega a recuperar en el registro).

    La resolución temporal de la latencia es 1/fs: con fs = 10 Hz, 0,1 s.
    """
    s, ds = f"{col}_suave", f"d{col}_dt"
    t = df["t_s"].to_numpy(dtype=float)
    A = df[s].to_numpy(dtype=float)
    dA = df[ds].to_numpy(dtype=float)
    fs = df.attrs.get("fs_hz") or 1.0 / np.median(np.diff(np.unique(t)))

    nan = float("nan")
    out = dict(t_estimulo_s=t_estimulo, basal=nan, minimo=nan, t_minimo_s=nan,
               amplitud=nan, amplitud_pct=nan,
               latencia_s=nan, latencia_tangente_s=nan,
               vel_constriccion_max=nan, vel_constriccion_media=nan,
               vel_redilatacion_max=nan, vel_redilatacion_media=nan,
               t_recuperacion_s=nan, recupera=False,
               frac_artefactos_ventana=nan, resolucion_temporal_s=1.0 / fs, advertencia="")

    # --- basal ---
    mb = (t >= t_estimulo - ventana_basal_s) & (t < t_estimulo) & ~np.isnan(A)
    if mb.sum() < 3:
        out["advertencia"] = "pocas muestras válidas en la ventana basal"
        return out
    basal = float(np.median(A[mb]))
    sigma = 1.4826 * _mad(A[mb])
    out["basal"] = basal

    # --- ventana de respuesta ---
    t_fin = t_estimulo + ventana_respuesta_s
    if t_siguiente is not None:
        t_fin = min(t_fin, t_siguiente)
    mr = (t >= t_estimulo) & (t <= t_fin)
    if "artefacto" in df:
        out["frac_artefactos_ventana"] = float(df["artefacto"].to_numpy()[mr].mean()) if mr.any() else nan
    ok = mr & ~np.isnan(A)
    if ok.sum() < 3:
        out["advertencia"] = "pocas muestras válidas tras el estímulo"
        return out

    idx = np.flatnonzero(ok)
    i_min = idx[np.argmin(A[idx])]
    minimo, t_min = float(A[i_min]), float(t[i_min])
    amplitud = basal - minimo
    out.update(minimo=minimo, t_minimo_s=t_min, amplitud=amplitud,
               amplitud_pct=100 * amplitud / basal)
    if amplitud <= 0:
        out["advertencia"] = "no hay constricción (el mínimo no baja del basal)"
        return out

    # --- latencia por umbral ---
    nivel = basal - max(k_latencia * sigma, frac_latencia * basal)
    t_ini = nan
    for i in idx:
        if i > i_min:
            break
        tramo = A[i:i + n_sostenido]
        if len(tramo) == n_sostenido and np.all(tramo < nivel) and not np.any(np.isnan(tramo)):
            if i > 0 and not np.isnan(A[i - 1]):
                t_ini = _cruce_lineal(t[i - 1], A[i - 1], t[i], A[i], nivel)
            else:
                t_ini = t[i]
            break
    if np.isfinite(t_ini):
        out["latencia_s"] = max(t_ini - t_estimulo, 0.0)

    # --- velocidad de constricción y latencia por tangente ---
    ini = np.isfinite(t_ini)
    mc = (t >= (t_ini if ini else t_estimulo)) & (t <= t_min) & ~np.isnan(dA)
    if mc.any():
        j = np.flatnonzero(mc)[np.argmin(dA[mc])]       # dA más negativo
        out["vel_constriccion_max"] = float(-dA[j])
        if dA[j] < 0:
            t_tan = t[j] + (basal - A[j]) / dA[j]
            out["latencia_tangente_s"] = max(float(t_tan - t_estimulo), 0.0)
    if ini and t_min > t_ini:
        out["vel_constriccion_media"] = amplitud / (t_min - t_ini)

    # --- redilatación ---
    fin_red = t.max() if t_siguiente is None else min(t.max(), t_siguiente)
    mrd = (t >= t_min) & (t <= fin_red) & ~np.isnan(dA)
    if mrd.any():
        out["vel_redilatacion_max"] = float(np.max(dA[mrd]))
    objetivo = minimo + fraccion_recuperacion * amplitud
    post = np.flatnonzero((t > t_min) & (t <= fin_red) & ~np.isnan(A) & (A >= objetivo))
    if post.size:
        k = post[0]
        t_rec = _cruce_lineal(t[k - 1], A[k - 1], t[k], A[k], objetivo) if not np.isnan(A[k - 1]) else t[k]
        out["t_recuperacion_s"] = float(t_rec)
        out["recupera"] = True
        if t_rec > t_min:
            out["vel_redilatacion_media"] = fraccion_recuperacion * amplitud / (t_rec - t_min)
    else:
        out["advertencia"] = (out["advertencia"] + " no se recupera "
                              f"{int(100 * fraccion_recuperacion)} % en el registro").strip()
    return out


def parametros_todos(df: pd.DataFrame, estimulos_s: list, col: str = "area_px", **kw) -> pd.DataFrame:
    """Aplica parametros_plr a cada estímulo (usa el siguiente como límite)."""
    est = sorted(estimulos_s)
    filas = []
    for i, te in enumerate(est):
        sig = est[i + 1] if i + 1 < len(est) else None
        p = parametros_plr(df, te, col=col, t_siguiente=sig, **kw)
        p["trial"] = i + 1
        filas.append(p)
    return pd.DataFrame(filas)


# =============================================================================
# 6. VARIABILIDAD ENTRE TRIALS / REGISTROS
# =============================================================================

def resumen_trials(tabla: pd.DataFrame,
                   cols=("latencia_s", "latencia_tangente_s", "amplitud", "amplitud_pct",
                         "vel_constriccion_max", "vel_redilatacion_max")) -> pd.DataFrame:
    """Media, desvío (ddof=1), CV (%) y n por parámetro. Con n < 2 el desvío es NaN."""
    filas = {}
    for c in cols:
        v = tabla[c].astype(float).dropna()
        m = v.mean() if len(v) else np.nan
        s = v.std(ddof=1) if len(v) > 1 else np.nan
        filas[c] = dict(n=len(v), media=m, desvio=s,
                        cv_pct=(100 * s / abs(m)) if (len(v) > 1 and m) else np.nan)
    return pd.DataFrame(filas).T


# =============================================================================
# 7. CURVA SINTÉTICA DE VERDAD CONOCIDA (para verificar el método)
# =============================================================================

def plr_sintetica(fs: float, dur: float = 12.0, t_est: float = 3.0, lat: float = 0.25,
                  A0: float = 300.0, Amin: float = 150.0, tau_c: float = 0.25,
                  tau_r: float = 1.5, ruido: float = 0.01, seed: int = 0):
    """Área sintética: basal A0, constricción exponencial desde t_est+lat, redilatación lenta.

    Sirve para comprobar que detectar_artefactos / limpiar_serie / parametros_plr
    recuperan latencia y amplitud conocidas. NO son datos reales.
    Devuelve (t, A).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    A = np.full_like(t, A0)
    t_on = t_est + lat
    t_min = t_on + 4 * tau_c
    for i, ti in enumerate(t):
        if t_on <= ti < t_min:
            A[i] = Amin + (A0 - Amin) * np.exp(-(ti - t_on) / tau_c)
        elif ti >= t_min:
            A[i] = A0 - (A0 - Amin) * np.exp(-(ti - t_min) / tau_r)
    A *= np.exp(rng.normal(0, ruido, A.size))
    return t, A
