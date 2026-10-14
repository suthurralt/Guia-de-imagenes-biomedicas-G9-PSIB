"""
src/modulo1/segmentacion.py — Módulo 2, punto 2.1
"Segmentacion y Morfologia Matematica de la Pupila"

Idea general
------------
Completar

"""

import cv2
import numpy as np
from src.modulo1.calidad_filtrado import filtro_gaussiano, conv_grises, aplicar_clahe
from src.modulo1.carga_muestreo import rgb_a_hsv
from src.config import cargar_config
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
 # Toma de entrada la imagen y el radio del disco que se quiere hallar. Como no 
 # se conoce este radio, estimar_pupila le va a pasar varios y se queda con el que más
 # contraste le da.

def _contraste_disco(img, r): 
    """Mapa de contraste para un radio r: brillo medio de un anillo exterior
    menos brillo medio de un disco interior. Si da un valor alto, se tiene un disco oscuro rodeado
    de algo más claro (iris sobre esclerótica, pupila sobre iris)."""
    #===========

    # 1) Se define el kernel.
    c = 2 * r # Índice del centro del kernel en píxeles.

    yy, xx = np.ogrid[:4 * r + 1, :4 * r + 1] # Crea dos coordenadas sin construir la grilla entera.
    # yy es una columna de 0 a 4r, xx una fila de 0 a 4r, y se expande solo si hace falta a (4r+1)(4r+1)

    #===========

    # 2) Distancia de cada píxel al centro del kernel.
    d = np.hypot(xx - c, yy - c) #==> matriz (4r+1)(4r+1), cada celda guarda la distancia de ese píxel al centro del kernel.
    
    #===========

    # 3) Definición de disco y anillo.
    disco = (d <= r).astype(np.float32) # 1 píxeles a distancia <= r del centro, 0 sino.
    # Define un círculo relleno.

    anillo = ((d > 1.15 * r) & (d <= 1.6 * r)).astype(np.float32) # 1 en píxeles a distancia entre 1.15r y 1.6r del centro, 0 sino.
    # Dibuja una corona alrededor del disco.

    #===========

    # 4) Normalización por la cantidad de 1s

    # Divide cada molde por la cantidad de 1s que tiene ==> c/ molde suma 1 y, ...
    # ... al aplicar el kernel sobre la imagen, el resultado en cada posición es un promedio.
    # Además permite comparar disco y anillo.
    disco /= disco.sum()
    anillo /= anillo.sum()

    #===========

    # 5) Padding
    # Se genera un paddeado de 2r píxeles relleno con un reflejo del borde de la imagen.
    pad = 2 * r
    P = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    #===========

    # 6) Operación diferencia anillo - disco y recorrido del kernel sobre la imagen. 
    # Desliza el kernel por cada píxel de la imagen y, en cada posición, calcula el promedio ponderado de lo que cae debajo.
    out = cv2.filter2D(P, -1, anillo) - cv2.filter2D(P, -1, disco) # Esta es la resta anillo - disco

    #===========

    #7) Salida
    # El resultado es alto donde el disco es oscuro y el anillo es claro.
    # Recorta padding para volver al tamaño original.
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
    #===========

    # 1) Preparación 
    h, w = img_gris.shape[:2] #==> alto y ancho de la imagen.
    img = cv2.GaussianBlur(img_gris, (5, 5), 0).astype(np.float32) #==> suavizado con Gaussiano 5x5 para que el ruido y los detalles finos no dominen el promedio.
                                                                   #==> conversión a float32 para _contraste_disco
    yy, xx = np.mgrid[:h, :w] #==> matrices del tamaño de la imagen (yy guarda fila de cada píxel, xx la columna).

    #===========
    # 2) Iris 

    mejor = None # Guarda mejor candidato de (contraste, x, y, r)
    # Límites del radio (10% y 17% del ancho)
    r_min = max(int(radio_iris_rel[0] * w), 4) # Evita radios muy chicos si la imagen es chica.
    r_max = max(int(radio_iris_rel[1] * w), r_min) # rmax > rmin

    # Iteración sobre distintos valores de radio r, de 2 en 2.
    for r in range(r_min, r_max + 1, 2):
        s = _contraste_disco(img, r) ## ==> Llamado a _contraste_disco

        # Se pone un valor muy grande y negativo en un marco de r/2 píxeles en los 4 bordes del mapa.
        # Es una forma de decir que el centro del iris nunca va a estar pegado al borde, donde se hizo un padding.
        m = int(r * 0.5)                   
        s[:m] = s[-m:] = -1e9
        s[:, :m] = s[:, -m:] = -1e9

        # Guarda el máximo del mapa de contraste s y su posición (x,y), el píxel de la imagen donde 
        # _contraste_disco dio el valor más alto para ese radio rl
        _, val, _, (x, y) = cv2.minMaxLoc(s)

        # Si este radio dio más contraste que el mejor radio hasta ahora, guardar: contraste, centro y radio.
        if mejor is None or val > mejor[0]:
            mejor = (val, x, y, r)

    # Obtiene el mejor. Si su contraste no llega a 25 niveles de grises, asume que no hay iris claro
    # (por ejemplo, está parpadeando) y devuelve None.
        
    val_iris, ix, iy, ir = mejor
    if val_iris < contraste_min:
        return None
    
    #===========
    # 3) Pupila dentro del iris 

    mejor_p = None # Guarda mejor candidato.

    # Límites del radio del la pupila (20% a 60% del radio del iris)
    rp_min = max(int(radio_pupila_rel[0] * ir), 3)
    rp_max = max(int(radio_pupila_rel[1] * ir), rp_min)

    # Itera con distintos radios en el rango elegido.
    for r in range(rp_min, rp_max + 1, 2):
        s = _contraste_disco(img, r)

        # Excluyo centros a menos de r/2 del borde: ahí el valor depende del relleno
        # por reflejo (padding).
        # Se supone que el centro del iris no está pegado al borde de la ROI.
        s[np.hypot(xx - ix, yy - iy) > 0.5 * ir] = -1e9

        # Busca máximo y su posición, compara radios, se queda con el ganador.
        _, val, _, (x, y) = cv2.minMaxLoc(s)
        if mejor_p is None or val > mejor_p[0]:
            mejor_p = (val, x, y, r)
    _, px, py, pr = mejor_p

    #===========
    # 4) Salida. Convierte todo a float y devuelve la pupila o la pupila con el iris.
    # Devuelve (cx, cy, r).

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
    #===========
    # 1) Círculo inicial: fila y columna, en ese orden.
    t = np.linspace(0, 2 * np.pi, 100)
    inicial = np.array([cy + factor * r * np.sin(t),
                        cx + factor * r * np.cos(t)]).T

    #===========

    # 2) Suavizado interno, y pasar la imagen a decimales entre 0 y 255
    # Esto evita que la curva se enganche en pestañas, reflejos, etc.
    # Se divide por 255 para normalizar los píxeles al rango [0.0, 1.0], requerido por la función de contornos activos.
    
    suave = filtro_gaussiano(img_gris,k,sigma) / 255.0

    #===========

    # 3) Ajustar la curva hacia los bordes oscuros de la pupila.
    # MUY importante que haya un LÍMITE de iteraciones: 500 en este caso.
    contorno = active_contour(suave, inicial, alpha=alpha, beta=beta,
                              gamma=gamma, w_line=0, w_edge=1,
                              max_num_iter=500)

    #===========

    # 4) Pasar el contorno a máscara: rellenar el polígono.
    # Se crea una imagen negra del mismo tamaño que la original.
    # Luego se pinta de blanco (255)  el interior del contorno y deja el resto en 0.


    mascara = np.zeros(img_gris.shape, dtype=np.uint8)
    filas, columnas = polygon(contorno[:,0], contorno[:, 1], shape=img_gris.shape)
    mascara[filas, columnas] = 255

    return mascara, contorno

    #===========

# =============================================================================
# 2.3 WATERSHED
# =============================================================================

# Creación de marcadores.
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
    #===========
    # 1) Alto, ancho de la imagen, coordenadas para cada píxel y distancia al centro

    alto, ancho = img_gris.shape

    Y, X = np.ogrid[:alto, :ancho]

    # Distancia de cada píxel al centro estimado (devuelve matriz del tamaño de la imagen donde cada celda contiene qué tan lejos
    # está ese píxel del centro).

    distancia = np.sqrt((X-cx)**2 + (Y-cy)**2)

    #===========
    # 2) Máscara del interior

    mascara_interior = distancia <= factor_interior*r

    #===========
    # 3) Máscara del anillo exterior

    mascara_exterior = (
        (distancia >= factor_anillo_interior * r) &
        (distancia <= factor_anillo_exterior *r)
    )

    #===========
    # 4) Inicializa toda la matriz en 0. Las zonas con valor 0 representan la región de incertidumbre.

    marcadores = np.zeros((alto, ancho), dtype=np.int32)

    #===========
    # 5) Asignar etiquetas al círculo interior y al anillo exterior.

    marcadores[mascara_interior] = 1
    marcadores[mascara_exterior] = 2
    # Devuelve una matriz que le dice al algoritmo de inundación con "agua de qué" rellenar,
    # y dónde no llenar, sino esperar que las dos aguas se encuentren para construir el dique.

    return marcadores


# Cálculo del gradiente con Sobel.

def calcular_gradiente_sobel(img_gris):
    """
    Calcula la magnitud del gradiente Sobel de una imagen en grises.

    Parámetros:
        img_gris: imagen en escala de grises.

    Retorna:
        Magnitud del gradiente Sobel.
    """

    # 1) Convertir la imagen de enteros de 8 bits a flotantes de 32 bits
    # Esto es para cambiar el rango dinámico y evitar error de overflow
    img_float = img_gris.astype(np.float32)

    #===========

    # 2) Calcular el gradiente horizontal
    grad_x = cv2.Sobel(
        img_float, cv2.CV_32F, 1, 0, ksize=3
    )

    #===========

    # 3) Calcular el gradiente vertical
    grad_y = cv2.Sobel(
        img_float, cv2.CV_32F, 0, 1, ksize=3
    )

    #===========
    # 4) Combinar ambas componentes para obtener la magnitud del gradiente
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
    #===========

    # 1) Calcular el gradiente Sobel (relieve)
    gradiente = calcular_gradiente_sobel(img_gris)

    #===========

    # 2) Aplicar Watershed usando los marcadores (inundación)
    etiquetas = watershed(
        gradiente,
        markers=marcadores)

    #===========

    # 3) Obtener únicamente la región de la pupila
    mascara = etiquetas == 1

    return mascara, etiquetas


# =============================================================================
# 2.4 K - means
# =============================================================================

# Se fija random_state para que los resultados sean reproducibles al ejecutar K-means con los mismos datos y parámetros.
# r, k y factor_ventana se encuentran en `config.yaml`

def segmentar_kmeans(roi, cx, cy, r, k, factor_ventana, random_state=42):
    """
    K-means sobre S y V en una ventana local alrededor de la pupila.
    Devuelve una máscara uint8 (0/255) del tamaño de la ROI.
    """
    alto, ancho = roi.shape[:2]

    # 1) Paso RGB a HSV
    hsv = rgb_a_hsv(roi)
    S = hsv["S"]
    V = hsv["V"]

    # 2) Defino ventana local (sin salirse de la imagen) 
    mitad = int(factor_ventana * r)
    x0 = max(int(cx) - mitad, 0)
    x1 = min(int(cx) + mitad, ancho)      
    y0 = max(int(cy) - mitad, 0)
    y1 = min(int(cy) + mitad, alto)      

    S_ven = S[y0:y1, x0:x1]
    V_ven = V[y0:y1, x0:x1]
    alto_v, ancho_v = S_ven.shape

    # 3) Características y K-means
    caracteristicas = np.stack([S_ven,V_ven], axis=-1)     
    caracteristicas = caracteristicas.reshape(-1, 2).astype(np.float32)

    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    etiquetas = modelo.fit_predict(caracteristicas).reshape(alto_v, ancho_v)

    # 4) Grupo de la pupila 
    # distancia de cada píxel de la ventana al centro (OJO: coordenadas de la ventana)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distancia = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    region_central = distancia <= r                 

    grupo_pupila = np.bincount(etiquetas[region_central]).argmax()

    # 5) Pegar en máscara de tamaño completo 
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = (etiquetas == grupo_pupila).astype(np.uint8) * 255   # ¿qué grupo?

    return mascara

# =============================================================================
# 2.5 PIPELINE COMPLETO: ROI -> MÁSCARA DE LA PUPILA (WATERSHED)
# =============================================================================
# Watershed es el método elegido en 2.1. Esta función junta todos sus pasos para
# que la usen igual el notebook de morfología (2.2) y el Módulo 3. Todos los
# parámetros salen de `config.yaml` (se pasa el dict `cfg` entero).

def segmentar_pupila(roi_rgb, cfg):
    """Segmenta la pupila en una ROI a color con Watershed.

    Pasos: grises -> CLAHE -> estimar_pupila -> marcadores -> Watershed sobre
    el gradiente Sobel -> máscara uint8.

    roi_rgb : ROI recortada, RGB uint8 (como la devuelven cm.extraer_frame y
              cm.recortar_roi).
    cfg     : diccionario de config.yaml. Se leen modulo1_2.clahe_clip,
              modulo1_2.clahe_grilla y modulo2.watershed.*

    Devuelve un diccionario:
        "gris"    : ROI en grises con CLAHE (uint8), la imagen sobre la que
                    se segmentó.
        "pupila"  : (cx, cy, r) estimados en píxeles, o None.
        "mascara" : máscara uint8 (255 = pupila, 0 = resto), o None.

    "pupila" y "mascara" son None cuando estimar_pupila no encuentra un iris
    con contraste suficiente (por ejemplo, en un parpadeo).
    """
    #===========
    # 1) Grises + CLAHE (parámetros del módulo 1.2)
    p = cfg["modulo1_2"]
    gris = aplicar_clahe(conv_grises(roi_rgb), p["clahe_clip"], p["clahe_grilla"])

    #===========
    # 2) Centro y radio aproximados de la pupila (None si no hay iris claro)
    pupila = estimar_pupila(gris)
    if pupila is None:
        return {"gris": gris, "pupila": None, "mascara": None}
    cx, cy, r = pupila

    #===========
    # 3) Marcadores y Watershed (parámetros de modulo2.watershed)
    w = cfg["modulo2"]["watershed"]
    marcadores = crear_marcadores(
        gris, cx, cy, r,
        w["factor_interior"], w["factor_anillo_interior"], w["factor_anillo_exterior"])
    mascara_bool, _ = segmentar_watershed(gris, marcadores)

    #===========
    # 4) bool -> uint8 (255 = pupila), el mismo formato que Otsu, Snake y K-means
    mascara = mascara_bool.astype(np.uint8) * 255

    return {"gris": gris, "pupila": pupila, "mascara": mascara}

