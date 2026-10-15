"""
src/modulo4/compresion.py — Módulo 4: Compresión de la información de imagen

Idea general
------------
Se comprime el frame original y las ROIs segmentadas con tres esquemas y se
mide cuánto se ahorra (ratio de compresión) y cuánto se pierde (PSNR/SSIM):

    - Huffman  (sin pérdida): códigos de longitud variable según la frecuencia
                              de cada nivel de gris. Implementación propia.
    - RLE      (sin pérdida): se guardan pares (valor, cantidad de repeticiones)
                              recorriendo la imagen en horizontal o en vertical.
    - JPEG     (con pérdida): DCT por bloques de 8x8 + cuantización según el
                              factor de calidad Q. Se usa el códec de OpenCV.

Además hay funciones para:
    - medir artefactos de bloque (blocking) de forma objetiva,
    - medir el error por zonas anatómicas (borde pupilar, iris, esclerótica),
    - estimar el diámetro pupilar sobre imágenes comprimidas con JPEG.

Convención de tamaños: la imagen sin comprimir ocupa 8 bits por muestra
(uint8). Ratio = bits originales / bits comprimidos  (ratio > 1 = ahorra).
"""

from __future__ import annotations

import heapq
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from src.modulo1 import calidad_filtrado as cf
from src.modulo1 import carga_muestreo as cm
from src.modulo2 import morfologia as morf
from src.modulo2 import segmentacion as seg
from src.modulo2.metricas import dice

BITS_POR_MUESTRA = 8  # uint8: cada muestra (píxel o canal) ocupa 1 byte


# =============================================================================
# 0. AUXILIARES
# =============================================================================

def bits_originales(img: np.ndarray) -> int:
    """Tamaño de la imagen sin comprimir, en bits (8 bits por muestra)."""
    return int(img.size * BITS_POR_MUESTRA)


def a_gris(img: np.ndarray) -> np.ndarray:
    """Devuelve la imagen en grises (si ya es 2D la deja igual)."""
    return img if img.ndim == 2 else cf.conv_grises(img)


def entropia(img: np.ndarray) -> float:
    """Entropía de Shannon de los niveles de la imagen, en bits/muestra.

        H = - sum_k p_k · log2(p_k)

    Es el mínimo teórico de bits por muestra que puede lograr un código
    sin pérdida que codifica cada muestra por separado (como Huffman).
    """
    conteo = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = conteo[conteo > 0] / conteo.sum()
    return float(-(p * np.log2(p)).sum())


# =============================================================================
# 1. HUFFMAN (sin pérdida)
# =============================================================================
# Se construye el árbol de Huffman a partir del histograma y se pasa a la forma
# "canónica": los códigos quedan determinados SOLO por la longitud de cada uno.
# Así, para decodificar alcanza con guardar 256 longitudes (la cabecera), y no
# hace falta guardar el árbol entero.

def huffman_longitudes(frecuencias: np.ndarray) -> np.ndarray:
    """Longitud del código de Huffman de cada símbolo (0..255).

    Algoritmo: se ponen todos los símbolos en una cola de prioridad ordenada
    por frecuencia; se juntan los dos menos frecuentes en un nodo nuevo (con
    la suma de sus frecuencias) hasta que queda uno solo. Cada vez que un
    símbolo queda "debajo" de una unión, su código crece un bit.

    Devuelve un array de 256 enteros (0 = el símbolo no aparece).
    """
    longitudes = np.zeros(256, dtype=np.int64)
    simbolos = np.flatnonzero(frecuencias)

    # Caso borde: imagen de un solo valor -> se le asigna un código de 1 bit.
    if len(simbolos) == 1:
        longitudes[simbolos[0]] = 1
        return longitudes

    # Cada elemento de la cola: (frecuencia, desempate, lista de símbolos).
    # El "desempate" evita que Python compare listas cuando hay frecuencias
    # iguales y hace que el resultado sea siempre el mismo.
    cola = [(int(frecuencias[s]), int(s), [int(s)]) for s in simbolos]
    heapq.heapify(cola)
    contador = 256
    while len(cola) > 1:
        f1, _, g1 = heapq.heappop(cola)
        f2, _, g2 = heapq.heappop(cola)
        for s in g1 + g2:          # todos los símbolos de ambos subárboles
            longitudes[s] += 1     # bajan un nivel -> +1 bit
        heapq.heappush(cola, (f1 + f2, contador, g1 + g2))
        contador += 1
    return longitudes


def huffman_canonico(longitudes: np.ndarray) -> np.ndarray:
    """Códigos canónicos (como enteros) a partir de las longitudes.

    Se ordenan los símbolos por (longitud, símbolo) y se numeran en binario:
    al pasar a una longitud mayor se agrega un 0 a la derecha (<<).
    """
    codigos = np.zeros(256, dtype=np.uint64)
    orden = sorted((int(l), s) for s, l in enumerate(longitudes) if l > 0)
    codigo, long_prev = 0, orden[0][0]
    for l, s in orden:
        codigo <<= (l - long_prev)
        codigos[s] = codigo
        codigo += 1
        long_prev = l
    return codigos


def huffman_comprimir(img: np.ndarray) -> dict:
    """Codifica la imagen con Huffman. Devuelve un "paquete" con el flujo de
    bits empaquetado en bytes y lo necesario para decodificar."""
    plano = img.ravel()
    frec = np.bincount(plano, minlength=256)
    longitudes = huffman_longitudes(frec)
    codigos = huffman_canonico(longitudes)

    # Escritura vectorizada del flujo de bits: cada muestra ocupa L bits a
    # partir de su posición "offset". Se escribe bit por bit de código (como
    # mucho max(L) pasadas), pero para todas las muestras a la vez.
    L = longitudes[plano]
    C = codigos[plano]
    offset = np.cumsum(L) - L
    n_bits = int(L.sum())
    bits = np.zeros(n_bits, dtype=np.uint8)
    for b in range(int(L.max())):
        sel = L > b
        bits[offset[sel] + b] = (C[sel] >> (L[sel] - 1 - b).astype(np.uint64)) & 1

    return {
        "flujo": np.packbits(bits),
        "n_bits": n_bits,
        "longitudes": longitudes,
        "forma": img.shape,
    }


def huffman_tamano_bits(paquete: dict, con_cabecera: bool = True) -> int:
    """Bits que ocupa lo comprimido. Cabecera = 256 longitudes de 8 bits +
    la forma de la imagen (3 enteros de 32 bits)."""
    cabecera = 256 * 8 + 3 * 32
    return paquete["n_bits"] + (cabecera if con_cabecera else 0)


def huffman_descomprimir(paquete: dict) -> np.ndarray:
    """Decodifica un paquete de huffman_comprimir y devuelve la imagen."""
    longitudes = paquete["longitudes"]
    codigos = huffman_canonico(longitudes)
    n = int(np.prod(paquete["forma"]))
    n_bits = paquete["n_bits"]
    l_max = int(longitudes.max())
    bits = np.unpackbits(paquete["flujo"])[:n_bits]

    if l_max <= 22:
        # Decodificación por TABLA: se mira una ventana de l_max bits; como el
        # código es libre de prefijo, esa ventana determina el símbolo y su
        # longitud. Se precalculan todas las ventanas de una vez.
        tabla_s = np.zeros(1 << l_max, dtype=np.uint8)
        tabla_l = np.zeros(1 << l_max, dtype=np.int64)
        for s in np.flatnonzero(longitudes):
            l = int(longitudes[s])
            ini = int(codigos[s]) << (l_max - l)
            fin = (int(codigos[s]) + 1) << (l_max - l)
            tabla_s[ini:fin] = s
            tabla_l[ini:fin] = l
        relleno = np.concatenate([bits, np.zeros(l_max, dtype=np.uint8)])
        ventanas = np.zeros(n_bits, dtype=np.int64)
        for j in range(l_max):
            ventanas = (ventanas << 1) | relleno[j:j + n_bits]
        ventanas = ventanas.tolist()
        tabla_s, tabla_l = tabla_s.tolist(), tabla_l.tolist()
        salida = bytearray(n)
        pos = 0
        for i in range(n):
            w = ventanas[pos]
            salida[i] = tabla_s[w]
            pos += tabla_l[w]
    else:
        # Decodificación bit a bit (más lenta; solo si hay códigos muy largos).
        dic = {(int(longitudes[s]), int(codigos[s])): s
               for s in np.flatnonzero(longitudes)}
        salida = bytearray(n)
        i, codigo, l = 0, 0, 0
        for b in bits.tolist():
            codigo = (codigo << 1) | b
            l += 1
            s = dic.get((l, codigo))
            if s is not None:
                salida[i] = s
                i += 1
                codigo, l = 0, 0
    return np.frombuffer(bytes(salida), dtype=np.uint8).reshape(paquete["forma"])


# =============================================================================
# 2. RUN-LENGTH ENCODING (sin pérdida)
# =============================================================================
# La imagen se "aplana" en un vector siguiendo un recorrido y se guardan pares
# (valor, longitud de la racha). Si la imagen es a color, se recorre cada canal
# por separado (primero todo R, después G, después B).
#   - horizontal: fila por fila (izquierda -> derecha, arriba -> abajo)
#   - vertical  : columna por columna (arriba -> abajo, izquierda -> derecha)
# La longitud se guarda con `bits_longitud` bits; una racha más larga que
# 2^bits_longitud - 1 se parte en varias.

def _a_canales(img: np.ndarray) -> np.ndarray:
    """(alto, ancho) o (alto, ancho, C) -> (C, alto, ancho)."""
    return img[None] if img.ndim == 2 else np.moveaxis(img, 2, 0)


def recorrer(img: np.ndarray, direccion: str) -> np.ndarray:
    """Aplana la imagen en el orden de recorrido pedido."""
    c = _a_canales(img)
    if direccion == "horizontal":
        return c.reshape(-1)
    if direccion == "vertical":
        return c.transpose(0, 2, 1).reshape(-1)
    raise ValueError("direccion debe ser 'horizontal' o 'vertical'.")


def deshacer_recorrido(vector: np.ndarray, forma: tuple, direccion: str) -> np.ndarray:
    """Inversa de `recorrer`."""
    alto, ancho = forma[:2]
    n_can = 1 if len(forma) == 2 else forma[2]
    if direccion == "horizontal":
        c = vector.reshape(n_can, alto, ancho)
    else:
        c = vector.reshape(n_can, ancho, alto).transpose(0, 2, 1)
    return c[0] if len(forma) == 2 else np.moveaxis(c, 0, 2)


def rle_comprimir(img: np.ndarray, direccion: str = "horizontal",
                  bits_longitud: int = 8) -> dict:
    """Codifica la imagen con RLE en la dirección pedida."""
    v = recorrer(img, direccion)
    max_racha = (1 << bits_longitud) - 1

    # Dónde empieza cada racha: posición 0 y cada lugar donde el valor cambia.
    inicios = np.concatenate([[0], np.flatnonzero(v[1:] != v[:-1]) + 1])
    longitudes = np.diff(np.concatenate([inicios, [v.size]]))
    valores = v[inicios]

    # Partir las rachas largas en trozos de como mucho max_racha.
    trozos = -(-longitudes // max_racha)               # ceil(L / max_racha)
    valores = np.repeat(valores, trozos)
    long_partidas = np.full(int(trozos.sum()), max_racha, dtype=np.int64)
    ultimos = np.cumsum(trozos) - 1
    long_partidas[ultimos] = longitudes - max_racha * (trozos - 1)

    return {
        "valores": valores.astype(np.uint8),
        "longitudes": long_partidas,
        "forma": img.shape,
        "direccion": direccion,
        "bits_longitud": bits_longitud,
    }


def rle_tamano_bits(paquete: dict, con_cabecera: bool = True) -> int:
    """Bits ocupados: cada par = 8 bits de valor + bits_longitud de racha.
    Cabecera = forma (3 x 32 bits) + 1 bit de dirección."""
    n_pares = len(paquete["valores"])
    cabecera = 3 * 32 + 1
    return n_pares * (8 + paquete["bits_longitud"]) + (cabecera if con_cabecera else 0)


def rle_descomprimir(paquete: dict) -> np.ndarray:
    v = np.repeat(paquete["valores"], paquete["longitudes"])
    return deshacer_recorrido(v, paquete["forma"], paquete["direccion"])


def longitud_media_racha(img: np.ndarray, direccion: str) -> float:
    """Longitud media de las rachas (sin partir). Ayuda a explicar el ratio."""
    v = recorrer(img, direccion)
    n_rachas = 1 + int(np.count_nonzero(v[1:] != v[:-1]))
    return v.size / n_rachas


# =============================================================================
# 3. JPEG (con pérdida)
# =============================================================================

def jpeg_comprimir(img: np.ndarray, calidad: int) -> np.ndarray:
    """Codifica con JPEG (OpenCV) y devuelve los bytes del archivo .jpg.

    La imagen de entrada es RGB (o grises); OpenCV espera BGR, por eso se
    convierte. El tamaño incluye las cabeceras del archivo (tablas de
    cuantización y de Huffman), igual que si se guardara en disco.
    """
    datos = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", datos, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        raise RuntimeError("OpenCV no pudo codificar el JPEG.")
    return buf


def jpeg_descomprimir(buf: np.ndarray, color: bool) -> np.ndarray:
    if color:
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def jpeg_ida_y_vuelta(img: np.ndarray, calidad: int) -> tuple[np.ndarray, int]:
    """Comprime y descomprime. Devuelve (imagen reconstruida, bits)."""
    buf = jpeg_comprimir(img, calidad)
    return jpeg_descomprimir(buf, color=img.ndim == 3), int(buf.size * 8)


# =============================================================================
# 4. EVALUACIÓN COMPARATIVA (ratio, bpp, PSNR, SSIM)
# =============================================================================

def _fila(nombre_img, metodo, parametro, img, rec, bits_comp):
    sin_perdida = bool(np.array_equal(img, rec))
    return {
        "imagen": nombre_img,
        "metodo": metodo,
        "parametro": parametro,
        "bytes_original": bits_originales(img) / 8,
        "bytes_comprimido": bits_comp / 8,
        "ratio": bits_originales(img) / bits_comp,
        # bits por muestra: 8 = sin compresión
        "bits_por_muestra": bits_comp / img.size,
        "psnr_db": cf.psnr(img, rec),          # inf si es idéntica
        "ssim": cf.ssim(img, rec),
        "sin_perdida": sin_perdida,
    }


def evaluar_compresion(imagenes: dict, calidades: list, bits_longitud_rle: int = 8,
                       verificar_huffman: bool = True) -> tuple[pd.DataFrame, dict]:
    """Aplica Huffman, RLE (H y V) y JPEG (cada Q) a cada imagen.

    imagenes : {nombre: imagen uint8}
    Devuelve (tabla, reconstrucciones) con reconstrucciones[(nombre, etiqueta)].
    """
    filas, recs = [], {}
    for nombre, img in imagenes.items():
        # --- Huffman ---
        paq = huffman_comprimir(img)
        rec = huffman_descomprimir(paq) if verificar_huffman else img
        fila = _fila(nombre, "Huffman", "-", img, rec, huffman_tamano_bits(paq))
        fila["entropia_bits"] = entropia(img)
        fila["long_media_codigo"] = paq["n_bits"] / img.size
        filas.append(fila)
        recs[(nombre, "Huffman")] = rec

        # --- RLE horizontal y vertical ---
        for direccion in ("horizontal", "vertical"):
            paq = rle_comprimir(img, direccion, bits_longitud_rle)
            rec = rle_descomprimir(paq)
            fila = _fila(nombre, f"RLE {direccion}", f"{bits_longitud_rle} bits",
                         img, rec, rle_tamano_bits(paq))
            fila["racha_media"] = longitud_media_racha(img, direccion)
            filas.append(fila)
            recs[(nombre, f"RLE {direccion}")] = rec

        # --- JPEG para cada factor de calidad ---
        for q in calidades:
            rec, bits = jpeg_ida_y_vuelta(img, q)
            filas.append(_fila(nombre, "JPEG", int(q), img, rec, bits))
            recs[(nombre, f"JPEG Q={q}")] = rec

    return pd.DataFrame(filas), recs


def barrido_jpeg(imagenes: dict, calidades: list, bloque: int = 8) -> tuple[pd.DataFrame, dict]:
    """JPEG con muchos Q para trazar las curvas PSNR/SSIM vs ratio.
    Incluye el índice de bloques (sección 5) de cada reconstrucción.
    Devuelve (tabla, {(nombre, q): imagen reconstruida})."""
    filas, recs = [], {}
    for nombre, img in imagenes.items():
        for q in calidades:
            rec, bits = jpeg_ida_y_vuelta(img, q)
            fila = _fila(nombre, "JPEG", int(q), img, rec, bits)
            fila["indice_bloques"] = indice_bloques(rec, bloque)
            filas.append(fila)
            recs[(nombre, int(q))] = rec
    return pd.DataFrame(filas), recs


# =============================================================================
# 5. ARTEFACTOS DE BLOQUE
# =============================================================================
# JPEG cuantiza cada bloque de 8x8 por separado; a Q bajo los bloques vecinos
# dejan de "empalmar" y aparece un salto de intensidad justo en los bordes de
# la grilla 8x8. Índice de bloques:
#     B = (diferencia media entre píxeles vecinos que CRUZAN un borde de bloque)
#         / (diferencia media entre píxeles vecinos DENTRO de un bloque)
# Sin artefactos B ≈ 1; con blocking B > 1.

def indice_bloques(img: np.ndarray, bloque: int = 8, origen: tuple = (0, 0)) -> float:
    """Índice de bloques sobre la luminancia.

    origen: (fila, columna) de la esquina superior izquierda de `img` dentro
            de la imagen que se comprimió. Hace falta cuando se mide sobre un
            recorte (la ROI), para saber dónde caen los bordes de la grilla.
    """
    g = a_gris(img).astype(np.float64)
    oy, ox = origen
    dh = np.abs(np.diff(g, axis=1))                 # vecinos horizontales
    dv = np.abs(np.diff(g, axis=0))                 # vecinos verticales
    borde_x = (np.arange(dh.shape[1]) + 1 + ox) % bloque == 0
    borde_y = (np.arange(dv.shape[0]) + 1 + oy) % bloque == 0
    cruzan = np.concatenate([dh[:, borde_x].ravel(), dv[borde_y, :].ravel()])
    dentro = np.concatenate([dh[:, ~borde_x].ravel(), dv[~borde_y, :].ravel()])
    return float(cruzan.mean() / max(dentro.mean(), 1e-9))


# =============================================================================
# 6. ERROR POR ZONAS: BORDE PUPILAR vs IRIS vs ESCLERÓTICA
# =============================================================================

def zonas_ojo(mascara: np.ndarray, iris: tuple, ancho_borde: int,
              factor_iris: float, esclerotica_rango: tuple,
              esclerotica_filas: tuple) -> dict:
    """Máscaras booleanas de tres zonas de la ROI.

    borde_pupila : franja de ±ancho_borde px alrededor del contorno de la
                   máscara de la pupila (borde nítido oscuro/claro).
    iris         : disco del iris (radio factor_iris · r_iris) sin la pupila
                   ni la franja del borde.
    esclerotica  : corona entre esclerotica_rango[0]·r_iris y
                   esclerotica_rango[1]·r_iris, solo a los costados del iris:
                   filas entre cy + esclerotica_filas[0]·r_iris y
                   cy + esclerotica_filas[1]·r_iris (y crece hacia abajo), para
                   no tomar párpados ni pestañas.
    """
    m = (mascara > 0).astype(np.uint8) * 255
    contorno = cv2.morphologyEx(
        m, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ) > 0
    dist_contorno = cv2.distanceTransform((~contorno).astype(np.uint8), cv2.DIST_L2, 5)
    borde = dist_contorno <= ancho_borde

    cx, cy, ri = iris
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    d_iris = np.hypot(xx - cx, yy - cy)

    iris_z = (d_iris <= factor_iris * ri) & ~borde & ~(m > 0)
    escl = ((d_iris >= esclerotica_rango[0] * ri) & (d_iris <= esclerotica_rango[1] * ri)
            & (yy >= cy + esclerotica_filas[0] * ri) & (yy <= cy + esclerotica_filas[1] * ri))
    return {"borde_pupila": borde, "iris": iris_z, "esclerotica": escl}


def textura_local(img: np.ndarray, ventana: int = 7) -> np.ndarray:
    """Desvío estándar local (en ventanas de ventana x ventana) de la
    luminancia. Mide cuánto varía la imagen "alrededor" de cada píxel:
    ~0 en zonas lisas, alto en bordes y texturas."""
    g = a_gris(img).astype(np.float64)
    media = cv2.blur(g, (ventana, ventana))
    return np.sqrt(np.maximum(cv2.blur(g * g, (ventana, ventana)) - media ** 2, 0))


def error_por_zona(original: np.ndarray, reconstruida: np.ndarray, zonas: dict,
                   ventana_textura: int = 7) -> dict:
    """RMSE (niveles de gris), SSIM medio y textura local del original en
    cada zona, sobre la luminancia.

    El SSIM se calcula como mapa (ventanas gaussianas σ=1,5, igual que en el
    Módulo 1) y se promedia dentro de cada zona. La textura es el desvío
    local medio del ORIGINAL: un mismo error se nota menos en una zona con
    mucha textura propia que en una lisa (efecto de enmascaramiento).
    """
    a = a_gris(original).astype(np.float64)
    b = a_gris(reconstruida).astype(np.float64)
    _, mapa = structural_similarity(a, b, data_range=255, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False, full=True)
    textura = textura_local(original, ventana_textura)
    salida = {}
    for nombre, z in zonas.items():
        salida[f"rmse_{nombre}"] = float(np.sqrt(np.mean((a[z] - b[z]) ** 2)))
        salida[f"ssim_{nombre}"] = float(mapa[z].mean())
        salida[f"textura_{nombre}"] = float(textura[z].mean())
    return salida


# =============================================================================
# 7. DIÁMETRO PUPILAR SOBRE IMÁGENES COMPRIMIDAS
# =============================================================================
# Se usa el pipeline de segmentación elegido en el Módulo 2 (Watershed,
# seg.segmentar_pupila) y, si está configurada, la operación morfológica final.
# Diámetro equivalente = diámetro del círculo de igual área: D = 2·sqrt(A/π).
# (Cuando el Módulo 3 tenga su propia función de diámetro, conviene usar esa.)

def diametro_equivalente(mascara: np.ndarray) -> tuple[float, int]:
    area = int(np.count_nonzero(mascara))
    return 2.0 * np.sqrt(area / np.pi), area


def medir_pupila(roi_rgb: np.ndarray, cfg: dict) -> dict:
    """Segmenta la pupila en la ROI y devuelve área, diámetro y máscara."""
    if roi_rgb.ndim == 2:                       # por si se trabaja en grises
        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_GRAY2RGB)
    res = seg.segmentar_pupila(roi_rgb, cfg)
    if res["mascara"] is None:
        return {"area_px": np.nan, "diametro_px": np.nan, "radio_estimado_px": np.nan,
                "mascara": None, "pupila": None}

    mascara = res["mascara"]
    m = cfg["modulo2"]["morfologia"]
    if m.get("operacion_final"):
        operacion = getattr(morf, m["operacion_final"])
        mascara = operacion(mascara, m["kernel"], m["iteraciones"], m["kernel_forma"])

    d, area = diametro_equivalente(mascara)
    return {"area_px": area, "diametro_px": d, "radio_estimado_px": res["pupila"][2],
            "mascara": mascara, "pupila": res["pupila"]}


def diametro_vs_calidad(frame_rgb: np.ndarray, cfg: dict, calidades: list) -> tuple[pd.DataFrame, dict]:
    """Diámetro pupilar en el frame original y tras comprimirlo con JPEG.

    Se comprime el FRAME COMPLETO (como lo haría una cámara o un video) y
    después se recorta la ROI y se segmenta.
    Devuelve (tabla, {etiqueta: resultado de medir_pupila}).
    """
    medidas = {"original": medir_pupila(cm.recortar_roi(frame_rgb, cfg["roi"]), cfg)}
    for q in calidades:
        rec, _ = jpeg_ida_y_vuelta(frame_rgb, q)
        medidas[f"Q={q}"] = medir_pupila(cm.recortar_roi(rec, cfg["roi"]), cfg)

    ref = medidas["original"]
    filas = []
    for etiqueta, med in medidas.items():
        fila = {"version": etiqueta,
                "calidad": np.nan if etiqueta == "original" else int(etiqueta[2:]),
                "area_px": med["area_px"], "diametro_px": med["diametro_px"],
                "radio_estimado_px": med["radio_estimado_px"]}
        fila["delta_diametro_px"] = med["diametro_px"] - ref["diametro_px"]
        fila["delta_diametro_pct"] = 100 * fila["delta_diametro_px"] / ref["diametro_px"]
        if med["mascara"] is not None and ref["mascara"] is not None:
            fila["dice_vs_original"] = dice(ref["mascara"], med["mascara"])
        filas.append(fila)
    return pd.DataFrame(filas), medidas


def diametro_en_video(ruta_video: str | Path, cfg: dict, calidades: list,
                      paso_frames: int = 1, tiempos: list | None = None) -> pd.DataFrame:
    """Diámetro pupilar a lo largo del video, original y con cada Q.

    Si `tiempos` es None se recorre todo el video tomando 1 de cada
    `paso_frames` frames; si se pasa una lista, solo esos instantes (s).
    Devuelve una tabla larga: t_s, version, diametro_px, area_px.
    """
    filas = []

    def procesar(frame_rgb, t):
        for version, img in [("original", frame_rgb)] + [
                (f"Q={q}", jpeg_ida_y_vuelta(frame_rgb, q)[0]) for q in calidades]:
            med = medir_pupila(cm.recortar_roi(img, cfg["roi"]), cfg)
            filas.append({"t_s": t, "version": version,
                          "diametro_px": med["diametro_px"], "area_px": med["area_px"]})

    if tiempos is not None:
        for t in tiempos:
            procesar(cm.extraer_frame(ruta_video, t), t)
        return pd.DataFrame(filas)

    cap = cv2.VideoCapture(str(ruta_video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % paso_frames == 0:
                procesar(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx / fps)
            idx += 1
    finally:
        cap.release()
    return pd.DataFrame(filas)


# =============================================================================
# 8. IMÁGENES A COMPRIMIR
# =============================================================================

def imagenes_de_trabajo(frame_rgb: np.ndarray, cfg: dict, raiz: Path) -> dict:
    """Arma el diccionario {nombre: imagen} que se va a comprimir.

    - "frame_original": el frame de `entrada` (config.yaml), en RGB o en
      grises según `modulo4.espacio`.
    - Una entrada por cada archivo de `modulo4.imagenes_segmentadas` (rutas
      relativas a la raíz del repo, p. ej. las del Módulo 2 en
      results/modulo2/). El nombre es el del archivo sin extensión. Se leen
      TAL CUAL están guardadas, en escala de grises.
    """
    c4 = cfg["modulo4"]
    frame = a_gris(frame_rgb) if c4["espacio"] == "gris" else frame_rgb
    imagenes = {"frame_original": np.ascontiguousarray(frame)}

    for ruta in c4["imagenes_segmentadas"]:
        p = Path(ruta) if Path(ruta).is_absolute() else Path(raiz) / ruta
        if not p.exists():
            raise FileNotFoundError(f"No encuentro la imagen segmentada: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"OpenCV no pudo leer: {p}")
        imagenes[p.stem] = img
    return imagenes
