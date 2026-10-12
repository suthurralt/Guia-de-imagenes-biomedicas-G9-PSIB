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

"""Convenciones de todo el archivo
-------------------------------
- Todas las funciones reciben y devuelven imágenes en ESCALA DE GRISES,
  matrices 2D de tipo uint8 (valores enteros 0..255).
- "r" = intensidad de entrada de un píxel, "s" = intensidad de salida.
  Una TRANSFORMACIÓN DE INTENSIDAD es una función s = T(r) que se aplica a
  cada píxel por separado, sin mirar a sus vecinos (operación "punto a punto").
- Un FILTRO ESPACIAL, en cambio, calcula cada píxel de salida mirando un
  vecindario (una ventanita, el "kernel") alrededor del píxel de entrada.
- Ninguna función modifica la imagen que recibe: siempre devuelve una nueva.
"""


# =============================================================================
# Utilidad común
# =============================================================================

def _a_uint8(img_float: np.ndarray) -> np.ndarray:
    """Lleva un array de números reales (float) al rango válido 0..255 y lo
    convierte a uint8.

    ¿Por qué hace falta?
      Las cuentas (sumar brillo, multiplicar contraste, agregar ruido) pueden
      dar valores menores que 0 o mayores que 255. Un uint8 NO puede guardar
      esos valores: si se convierte directo, "da la vuelta" (300 -> 44,
      -10 -> 246) y aparecen píxeles absurdos sin ningún aviso.
      np.clip "recorta": todo lo < 0 pasa a 0 y todo lo > 255 pasa a 255.
      Esto se llama SATURACIÓN: la información que quedaba afuera se pierde.

    El guion bajo al principio del nombre (_a_uint8) es una convención de
    Python: "función interna, de uso dentro de este archivo".
    """
    # np.round redondea al entero más cercano (si no, astype truncaría: 2.9 -> 2).
    return np.clip(np.round(img_float), 0, 255).astype(np.uint8)


def porcentaje_saturado(img: np.ndarray) -> tuple[float, float]:
    """Porcentaje de píxeles que quedaron "clavados" en 0 (negro puro) y en
    255 (blanco puro). Sirve para cuantificar cuánta información se perdió
    por saturación después de una transformación.

    (img == 0) da una matriz de True/False. .mean() trata True como 1 y False
    como 0, así que el promedio es directamente la FRACCIÓN de píxeles True.
    """
    return 100 * float((img == 0).mean()), 100 * float((img == 255).mean())


# =============================================================================
# 1.2.1 — Brillo y contraste:  s = a·r + b
# =============================================================================
#   b (brillo)    -> suma una constante: todo el histograma se DESPLAZA.
#   a (contraste) -> multiplica: el histograma se ESTIRA (a > 1) o comprime
#                    (a < 1). Si se estira más allá de 255, se "hiper-extiende":
#                    lo que se pasa queda saturado en 255.

def ajustar_brillo(img: np.ndarray, b: float) -> np.ndarray:
    """Suma la constante b a todos los píxeles: s = r + b."""
    # astype(np.float32): pasamos a números reales ANTES de sumar, para que
    # 230 + 50 = 280 no "dé la vuelta" en uint8. Después recortamos a 0..255.
    return _a_uint8(img.astype(np.float32) + b)


def ajustar_contraste(img: np.ndarray, a: float) -> np.ndarray:
    """Multiplica todos los píxeles por a: s = a·r.

    Ojo: multiplicar por 1,5 también aclara la imagen (un píxel de 100 pasa a
    150). Las diferencias entre píxeles crecen (100 y 110 -> 150 y 165:
    diferencia 10 -> 15), eso es "más contraste"."""
    return _a_uint8(img.astype(np.float32) * a)


# =============================================================================
# 1.2.2 — Escalado por rango de interés [f1, f2]
# =============================================================================
#
#   s = 255 · (r − f1) / (f2 − f1)      para f1 ≤ r ≤ f2
#   s = 0                                para r < f1
#   s = 255                              para r > f2
#
#   Es una RECTA que manda f1 -> 0 (negro) y f2 -> 255 (blanco). Todo el
#   rango de salida 0..255 se dedica SOLO a las intensidades de la pupila e
#   iris; lo demás (piel, esclerótica) se satura en blanco a propósito.
#   Pendiente = 255 / (f2 − f1): con f1=0, f2=100 la pendiente es 2,55, o sea
#   las diferencias de gris dentro del iris se agrandan 2,55 veces.

def rango_automatico(img: np.ndarray, p_bajo: float, p_alto: float) -> tuple[int, int]:
    """Propone [f1, f2] a partir de PERCENTILES de la imagen.

    El percentil p es el valor por debajo del cual está el p % de los píxeles.
    Ej.: percentil 25 = 100 significa "el 25 % de los píxeles vale menos que 100".
    En una ROI centrada en el ojo, la pupila y el iris son la parte más oscura,
    así que [p1, p25] suele caer sobre ellos. Se usa solo si en el config
    f1/f2 están en null (por ejemplo, con otra foto y sin ganas de mirar el
    histograma).
    """
    f1 = int(np.percentile(img, p_bajo))
    f2 = int(np.percentile(img, p_alto))
    return f1, f2


def escalar_rango(img: np.ndarray, f1: int, f2: int) -> np.ndarray:
    """Aplica el escalado lineal del intervalo [f1, f2] a [0, 255]."""
    if f2 <= f1:
        # Si f2 == f1 dividiríamos por cero; si f2 < f1 la recta "se da vuelta".
        raise ValueError(f"Se necesita f1 < f2 (llegó f1={f1}, f2={f2}).")
    r = img.astype(np.float32)
    # La misma fórmula para todos los píxeles a la vez (NumPy opera sobre la
    # matriz entera). Los que quedan < 0 o > 255 los recorta _a_uint8.
    return _a_uint8(255.0 * (r - f1) / (f2 - f1))


# =============================================================================
# 1.2.3 — Ecualización de histograma (global) y CLAHE (local)
# =============================================================================
#
# Idea: buscar una T(r) que deje el histograma lo más "plano" posible, es
# decir, que todas las intensidades se usen más o menos la misma cantidad de
# veces. La T que lo logra es la FUNCIÓN DE DISTRIBUCIÓN ACUMULADA (CDF):
#
#   CDF(k) = (cantidad de píxeles con intensidad ≤ k) / (total de píxeles)
#   s = round( 255 · (CDF(r) − CDF_min) / (1 − CDF_min) )
#
# CDF_min es la CDF en la intensidad más baja presente; restarla hace que el
# píxel más oscuro termine en 0 (si no, nunca llegaríamos al negro puro).
#
# Intuición: donde el histograma tiene MUCHOS píxeles (un pico), la CDF sube
# rápido -> la T tiene mucha pendiente -> esos niveles se separan mucho
# (más contraste). Donde hay pocos píxeles, la CDF es casi plana -> esos
# niveles se juntan.

def ecualizar_global(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ecualización global implementada "a mano" (para ver cada paso).

    Devuelve
    --------
    (imagen_ecualizada, lut)
        lut = "Look-Up Table" (tabla de búsqueda): vector de 256 posiciones
        donde lut[r] es el valor de salida para la intensidad r. Es la
        función T(r) en forma de tabla; la devolvemos para poder graficarla.
    """
    # 1) Histograma: np.bincount cuenta cuántas veces aparece cada entero.
    #    .ravel() aplana la matriz 2D a un vector 1D (bincount necesita 1D).
    #    minlength=256 asegura 256 posiciones aunque falten intensidades altas.
    hist = np.bincount(img.ravel(), minlength=256)

    # 2) Acumulada: cumsum hace sumas parciales [h0, h0+h1, h0+h1+h2, ...].
    #    Dividiendo por el total queda entre 0 y 1 (es una probabilidad).
    cdf = hist.cumsum() / img.size

    # 3) CDF_min: primer valor NO nulo de la CDF (= fracción de píxeles que
    #    tienen la intensidad más baja presente).
    cdf_min = cdf[cdf > 0][0]

    # 4) La transformación como tabla (lut), estirada a 0..255.
    lut = np.round(255 * (cdf - cdf_min) / (1 - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # 5) Aplicar la tabla: lut[img] es "indexado avanzado" de NumPy: para cada
    #    píxel con valor r, toma lut[r]. Transforma toda la imagen de una vez.
    return lut[img], lut


def aplicar_clahe(img: np.ndarray, clip_limit: float, grilla: int) -> np.ndarray:
    """CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Dos diferencias con la ecualización global:
    1. ADAPTATIVA (local): divide la imagen en una grilla de grilla×grilla
       bloques ("tiles") y ecualiza CADA bloque con su propio histograma.
       Así una zona oscura (la pupila) se realza según SU vecindario, no
       según la imagen entera dominada por la piel. Para que no se noten los
       bordes entre bloques, interpola bilinealmente entre bloques vecinos.
    2. CONTRASTE LIMITADO: antes de calcular la CDF, recorta los picos del
       histograma de cada bloque en clip_limit (y reparte el exceso entre
       todos los niveles). Un pico alto daría una pendiente enorme en la T y
       amplificaría el RUIDO de zonas uniformes; limitar el pico limita esa
       pendiente. clip_limit más alto = más contraste y más ruido.
    """
    # createCLAHE crea un "objeto" configurado; .apply() lo aplica a la imagen.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grilla, grilla))
    return clahe.apply(img)


# =============================================================================
# 1.2.4 — Ruido
# =============================================================================
# Todas reciben `rng`, un generador de números aleatorios de NumPy creado con
# una SEMILLA fija (np.random.default_rng(semilla)). Con la misma semilla, el
# ruido "aleatorio" sale idéntico en cada ejecución: los resultados del
# informe son reproducibles.

def ruido_gaussiano(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido ADITIVO gaussiano: s = r + n, con n ~ Normal(media 0, desvío sigma).

    Modela el ruido electrónico del sensor (térmico, de lectura): afecta a
    TODOS los píxeles un poquito, sin importar su brillo.
    rng.normal(0, sigma, size) genera una matriz del mismo tamaño que la
    imagen con valores al azar de esa distribución.
    """
    n = rng.normal(0.0, sigma, img.shape)
    return _a_uint8(img.astype(np.float32) + n)


def ruido_sal_pimienta(img: np.ndarray, densidad: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido IMPULSIVO: una fracción `densidad` de píxeles se reemplaza por
    0 (pimienta) o 255 (sal), mitad y mitad. El resto queda intacto.

    Modela píxeles muertos/quemados o errores de transmisión: pocos píxeles,
    pero con error enorme.
    """
    salida = img.copy()                 # copia: no tocamos la original
    u = rng.random(img.shape)           # números al azar uniformes en [0, 1)
    # Si u < densidad/2 -> pimienta. Si está entre densidad/2 y densidad -> sal.
    # Como u es uniforme, cada caso ocurre en (densidad/2) de los píxeles.
    salida[u < densidad / 2] = 0
    salida[(u >= densidad / 2) & (u < densidad)] = 255   # & = "y" elemento a elemento
    return salida


def ruido_speckle(img: np.ndarray, varianza: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido MULTIPLICATIVO (speckle): s = r + r·n = r·(1 + n), n ~ Normal(0, √varianza).

    A diferencia del gaussiano, el error es PROPORCIONAL al brillo: en zonas
    oscuras (pupila) casi no se nota, en zonas claras (piel) mucho. Es el
    ruido típico de ecografía, OCT o láser (interferencia de ondas).
    El desvío es la raíz de la varianza: np.sqrt(varianza).
    """
    r = img.astype(np.float32)
    n = rng.normal(0.0, np.sqrt(varianza), img.shape)
    return _a_uint8(r + r * n)


# =============================================================================
# 1.2.4 — Filtros espaciales
# =============================================================================
# Los tres recorren la imagen con una ventana de k×k píxeles centrada en cada
# píxel. k tiene que ser IMPAR (3, 5, 7...) para que la ventana tenga centro.

def filtro_media(img: np.ndarray, k: int) -> np.ndarray:
    """Reemplaza cada píxel por el PROMEDIO de su ventana k×k.

    Es una convolución con un kernel donde todos los pesos valen 1/k².
    Reduce bien el ruido gaussiano (promediar N valores con ruido independiente
    divide su desvío por √N), pero también promedia a través de los BORDES:
    el contorno de la pupila se desenfoca. Contra sal y pimienta es malo:
    un 255 aislado "contamina" a todos sus vecinos (queda una manchita gris).
    """
    return cv2.blur(img, (k, k))


def filtro_mediana(img: np.ndarray, k: int) -> np.ndarray:
    """Reemplaza cada píxel por la MEDIANA de su ventana k×k: ordena los k²
    valores y toma el del medio.

    NO es lineal (no es una convolución). Un valor extremo (0 o 255 de la
    sal y pimienta) queda en una punta de la lista ordenada y nunca es el del
    medio -> lo elimina por completo. Además preserva mejor los bordes: si la
    ventana está mitad en la pupila y mitad en el iris, la mediana es un
    valor de uno de los dos lados, no un promedio intermedio.
    """
    return cv2.medianBlur(img, k)


def filtro_gaussiano(img: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Promedio PONDERADO con pesos de una campana de Gauss de desvío sigma:
    el píxel central pesa más y los alejados, menos.

    Suaviza de forma más "natural" que la media (sin el efecto de caja) y es
    el filtro pasa-bajos ideal en el sentido de que su transformada de
    Fourier también es una gaussiana (sin rebotes) -> se retoma en 1.3.
    """
    return cv2.GaussianBlur(img, (k, k), sigma)


# =============================================================================
# Métricas de calidad: comparar una imagen procesada contra la original
# =============================================================================

def psnr(referencia: np.ndarray, img: np.ndarray) -> float:
    """PSNR = Peak Signal-to-Noise Ratio, en decibeles (dB).

        MSE  = promedio de (referencia − img)²       (error cuadrático medio)
        PSNR = 10 · log10( 255² / MSE )

    255 es el valor máximo posible ("pico") de la señal. Más dB = más parecida.
    Orientativo: > 30 dB muy parecida, 20–30 dB diferencias visibles,
    < 20 dB muy degradada. Si las imágenes son idénticas, MSE = 0 -> infinito.
    """
    # float64 para que el cuadrado de las diferencias no desborde.
    mse = np.mean((referencia.astype(np.float64) - img.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(referencia: np.ndarray, img: np.ndarray) -> float:
    """SSIM = Structural SIMilarity index (Wang et al., 2004). Va de −1 a 1;
    1 = idénticas.

    El PSNR solo mira diferencias píxel a píxel; el SSIM intenta parecerse a
    cómo percibimos: compara, en ventanitas locales, tres cosas:
      - luminancia (medias μ),
      - contraste (desvíos σ),
      - estructura (correlación σ_xy entre ambas imágenes).

        SSIM = [(2·μx·μy + C1)(2·σxy + C2)] / [(μx² + μy² + C1)(σx² + σy² + C2)]

    Las medias locales se calculan con un filtro gaussiano de 11×11 y σ=1,5
    (los valores del paper original). C1 y C2 son constantes chicas que
    evitan dividir por cero en zonas planas. Se devuelve el promedio del
    mapa SSIM de toda la imagen.
    """
    x = referencia.astype(np.float64)
    y = img.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Función auxiliar: "promedio local" = desenfoque gaussiano 11×11.
    def promedio_local(a):
        return cv2.GaussianBlur(a, (11, 11), 1.5)

    mu_x, mu_y = promedio_local(x), promedio_local(y)
    # Varianza = E[x²] − (E[x])² ; covarianza = E[xy] − E[x]·E[y]
    var_x = promedio_local(x * x) - mu_x ** 2
    var_y = promedio_local(y * y) - mu_y ** 2
    cov_xy = promedio_local(x * y) - mu_x * mu_y

    mapa = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return float(mapa.mean())
