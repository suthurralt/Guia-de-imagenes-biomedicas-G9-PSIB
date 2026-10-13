"""
src/modulo1/segmentacion.py — Módulo 2, punto 2.1
"Segmentacion y Morfologia Matematica de la Pupila"

Idea general
------------
Completar

"""
import cv2
import numpy as np
from src.modulo1.calidad_filtrado import filtro_gaussiano
from src.modulo1.carga_muestreo import rgb_a_hsv
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# =============================================================================
# 2.1 OTSU
# =============================================================================

def segmentar_otsu(img_gris):
    """Segmenta la pupila con umbralización automática de Otsu.

    Recibe una imagen en grises (alto, ancho), uint8.
    Devuelve (umbral, mascara): el umbral elegido por Otsu y una máscara
    binaria (255 = pupila, 0 = resto).

    cv2.threshold necesita saber qué hacer con cada píxel y cómo elegir el umbral. Eso se indica en el cuarto argumento:
    THRESH_BINARY_INV: los píxeles por debajo del umbral pasan a 255 y los de arriba, a 0.
    THRESH_OTSU: cálculo del umbral con Otsu (minimizando varianza intraclase y maximizando varianza interclase)

    """
    umbral, mascara = cv2.threshold(img_gris,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return umbral, mascara

# =============================================================================
# 2.2 ESTIMAR CENTRO DE PUPILA Y RADIO
# =============================================================================
 
def _contraste_disco(img, r):
    """Mapa de contraste para un radio r: brillo medio de un anillo exterior
    menos brillo medio de un disco interior. Valor alto = disco oscuro rodeado
    de algo más claro (iris sobre esclerótica, pupila sobre iris)."""
    c = 2 * r
    yy, xx = np.ogrid[:4 * r + 1, :4 * r + 1]
    d = np.hypot(xx - c, yy - c)
    disco = (d <= r).astype(np.float32)
    anillo = ((d > 1.15 * r) & (d <= 1.6 * r)).astype(np.float32)
    disco /= disco.sum()
    anillo /= anillo.sum()
    pad = 2 * r
    P = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    out = cv2.filter2D(P, -1, anillo) - cv2.filter2D(P, -1, disco)
    return out[pad:-pad, pad:-pad]
 
 
def estimar_pupila(img_gris, radio_iris_rel=(0.10, 0.17),
                   radio_pupila_rel=(0.20, 0.60), contraste_min=25,
                   devolver_iris=False):
    """Estima centro y radio de la pupila en una imagen en escala de grises.
 
    Etapa 1: busca el iris (disco oscuro grande). Acotar el radio evita que
             gane una sombra o un grupo de pestañas.
    Etapa 2: busca la pupila (disco oscuro chico) solo cerca del centro del iris.
             Funciona aunque un reflejo tape parte de la pupila.
 
    radio_iris_rel   : rango del radio del iris, fracción del ANCHO de la imagen.
    radio_pupila_rel : rango del radio de la pupila, fracción del radio del iris.
    contraste_min    : si el iris no destaca al menos esto (niveles de gris),
                       se devuelve None (p. ej. parpadeo).
    devolver_iris    : si True devuelve ((cx, cy, r_pupila), (cx, cy, r_iris)).
 
    Devuelve (cx, cy, r) en píxeles, o None.
    """
    h, w = img_gris.shape[:2]
    img = cv2.GaussianBlur(img_gris, (5, 5), 0).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
 
    # --- 1) Iris ---
    mejor = None
    r_min = max(int(radio_iris_rel[0] * w), 4)
    r_max = max(int(radio_iris_rel[1] * w), r_min)
    for r in range(r_min, r_max + 1, 2):
        s = _contraste_disco(img, r)
        m = int(r * 0.5)                       # centro lejos del borde
        s[:m] = s[-m:] = -1e9
        s[:, :m] = s[:, -m:] = -1e9
        _, val, _, (x, y) = cv2.minMaxLoc(s)
        if mejor is None or val > mejor[0]:
            mejor = (val, x, y, r)
    val_iris, ix, iy, ir = mejor
    if val_iris < contraste_min:
        return None
 
    # --- 2) Pupila dentro del iris ---
    mejor_p = None
    rp_min = max(int(radio_pupila_rel[0] * ir), 3)
    rp_max = max(int(radio_pupila_rel[1] * ir), rp_min)
    for r in range(rp_min, rp_max + 1, 2):
        s = _contraste_disco(img, r)
        s[np.hypot(xx - ix, yy - iy) > 0.5 * ir] = -1e9
        _, val, _, (x, y) = cv2.minMaxLoc(s)
        if mejor_p is None or val > mejor_p[0]:
            mejor_p = (val, x, y, r)
    _, px, py, pr = mejor_p
 
    pupila = (float(px), float(py), float(pr))
    if devolver_iris:
        return pupila, (float(ix), float(iy), float(ir))
    return pupila
# =============================================================================
# 2.2 SNAKE
# =============================================================================

def segmentar_snake(img_gris, cx, cy, r, 
                    factor,k,sigma,alpha,beta,gamma):
    """Segmenta la pupila con un contorno activo (snake).

    Parte de un círculo de radio `factor * r` centrado en (cx, cy).
    Devuelve (mascara, contorno): máscara binaria (255 = pupila) y los
    puntos finales de la curva, como array (N, 2) de (fila, columna).
    """
    # 1) Círculo inicial: fila y columna, en ese orden
    t = np.linspace(0, 2 * np.pi, 100)
    inicial = np.array([cy + factor * r * np.sin(t),
                        cx + factor * r * np.cos(t)]).T

    # 2) Suavizado interno, y pasar la imagen a decimales entre 0 y 255
    # Esto evita que la curva se enganche en pestañas, reflejos, etc.
    # Se divide por 255 para normalizar los píxeles al rango [0.0, 1.0], requerido
    # por la función de contornos activos.
    
    suave = filtro_gaussiano(img_gris,k,sigma) / 255.0

    # 3) Ajustar la curva hacia los bordes oscuros de la pupila.
    # MUY importante que haya un LÍMITE de iteraciones: 500 en este caso.
    contorno = active_contour(suave, inicial, alpha=alpha, beta=beta,
                              gamma=gamma, w_line=0, w_edge=1,
                              max_num_iter=500)

    # 4) Pasar el contorno a máscara: rellenar el polígono.
    # Se crea una imagen negra del mismo tamaño que la original.
    # Luego se pinta de blanco (255)  el interior del contorno y deja el resto en 0.

    mascara = np.zeros(img_gris.shape, dtype=np.uint8)
    filas, columnas = polygon(contorno[:,0], contorno[:, 1], shape=img_gris.shape)
    mascara[filas, columnas] = 255

    return mascara, contorno


# =============================================================================
# 2.3 WATERSHED
# =============================================================================
## Creación de marcadores.
def crear_marcadores(img_gris, cx, cy, r,
                     factor_interior,
                     factor_anillo_interior,
                     factor_anillo_exterior):
    """
    Crea tres zonas bien definidas dentro de la imagen basándose en la distancia
    euclídea de cada píxel al centro estimado de la pupila.
    factor_interior ==> es un porcentaje del radio, donde se sabe que los píxeles son de la pupila.
    factor_anillo_interior y factor_anillo_ exterior ==> definen un anillo rodeando la pupila,
    Al estar fuera de la pupila, hay certeza absoluta de que pertenecen al fondo.
    """
    alto, ancho = img_gris.shape

    # Crear coordenadas para cada píxel
    Y, X = np.ogrid[:alto, :ancho]

    # Distancia de cada píxel al centro estimado (devuelve matriz del tamaño de la imagen donde cada celda contoene qué tan lejos
    # está ese píxel del centro).
    distancia = np.sqrt((X-cx)**2 + (Y-cy)**2)

    # Máscara del interior
    mascara_interior = distancia <= factor_interior*r

    # Máscara del anillo exterior
    mascara_exterior = (
        (distancia >= factor_anillo_interior * r) &
        (distancia <= factor_anillo_exterior *r)
    )

    # Inicializa toda la matriz en 0. Las zonas con valor 0 representan la región de incertidumbre.
    marcadores = np.zeros((alto, ancho), dtype=np.int32)

    # Asignar etiquetas al círculo interior y al anillo exterior.
    marcadores[mascara_interior] = 1
    marcadores[mascara_exterior] = 2
    # Devuelve una matriz que le dice al algoritmo de inundación con "agua de qué" rellenar,
    # y dónde no llenar, sino esperar que las dos aguas se encuentren para construir el dique.

    return marcadores

## Cálculo del gradiente con Sobel.
def calcular_gradiente_sobel(img_gris):
    """
    Calcula la magnitud del gradiente Sobel de una imagen en grises.

    Parámetros:
        img_gris: imagen en escala de grises.

    Retorna:
        Magnitud del gradiente Sobel.
    """

    # Convertir la imagen de enteros de 8 bits a flotantes de 32 bits.
    # Esto es para cambiar el rango dinámico y evitar error de overflow.
    img_float = img_gris.astype(np.float32)

    # Calcular el gradiente horizontal
    grad_x = cv2.Sobel(
        img_float, cv2.CV_32F, 1, 0, ksize=3
    )

    # Calcular el gradiente vertical
    grad_y = cv2.Sobel(
        img_float, cv2.CV_32F, 0, 1, ksize=3
    )

    # Combinar ambas componentes para obtener la magnitud del gradiente
    gradiente = np.sqrt(grad_x**2 + grad_y**2)

    return gradiente


## Segmentación sobre gradiente Sobel usando marcadores
def segmentar_watershed(img_gris, marcadores):
    """
    Segmenta una imagen utilizando Watershed
    sobre el gradiente Sobel.

    Parámetros:
        img_gris: imagen en grises.
        marcadores: matriz con las etiquetas iniciales.

    Retorna:
        mascara: máscara binaria de la pupila.
        etiquetas: resultado completo de Watershed.
    """

    # 1. Calcular el gradiente Sobel (relieve)
    gradiente = calcular_gradiente_sobel(img_gris)

    # 2. Aplicar Watershed usando los marcadores (inundación)
    etiquetas = watershed(
        gradiente,
        markers=marcadores
    )

    # 3. Obtener únicamente la región de la pupila
    mascara = etiquetas == 1

    return mascara, etiquetas


# =============================================================================
# 2.4 K - means
# =============================================================================
def segmentar_kmeans(roi, cx, cy, r, k, factor_ventana=3, random_state=42):
    """
    K-means sobre S y V en una ventana local alrededor de la pupila.
    Devuelve una máscara uint8 (0/255) del tamaño de la ROI.
    """
    alto, ancho = roi.shape[:2]

    # Paso RGB a HSV
    hsv = rgb_a_hsv(roi)
    S = hsv["S"]
    V = hsv["V"]

    # Defino ventana local (sin salirse de la imagen) 
    mitad = int(factor_ventana * r)
    x0 = max(int(cx) - mitad, 0)
    x1 = min(int(cx) + mitad, ancho)      
    y0 = max(int(cy) - mitad, 0)
    y1 = min(int(cy) + mitad, alto)      

    S_ven = S[y0:y1, x0:x1]
    V_ven = V[y0:y1, x0:x1]
    alto_v, ancho_v = S_ven.shape

    # Características y K-means
    caracteristicas = np.stack([S_ven,V_ven], axis=-1)     
    caracteristicas = caracteristicas.reshape(-1, 2).astype(np.float32)

    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    etiquetas = modelo.fit_predict(caracteristicas).reshape(alto_v, ancho_v)

    # Grupo de la pupila ---
    # distancia de cada píxel de la ventana al centro (OJO: coordenadas de la ventana)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distancia = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    region_central = distancia <= r                 

    grupo_pupila = np.bincount(etiquetas[region_central]).argmax()

    # Pegar en máscara de tamaño completo 
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = (etiquetas == grupo_pupila).astype(np.uint8) * 255   # ¿qué grupo?

    return mascara

