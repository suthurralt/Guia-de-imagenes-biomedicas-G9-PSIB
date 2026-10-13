"""Funciones auxiliares del punto 1.3: bordes y dominio frecuencial.

Extraídas de 01_3_bordes_frecuencia.ipynb. Las funciones conservan el
procesamiento original; mostrar y zoom reciben explícitamente los datos
que antes obtenían de variables globales del notebook.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def a_0_255(im, percentil=99.5):
    '''Lleva cualquier matriz (gradientes, valores negativos, números enormes de
    la FFT...) al rango 0..255 SOLO PARA MOSTRARLA.

    Usa el percentil 99.5 como "blanco" en lugar del máximo: así un par de
    píxeles muy intensos (por ejemplo el reflejo del flash) no hacen que todo
    lo demás se vea negro. Lo que supera ese valor queda en 255.
    '''
    im = np.abs(im).astype(np.float64)          # valor absoluto: bordes "negativos" también cuentan
    tope = np.percentile(im, percentil)
    if tope == 0:                               # imagen toda en cero: evitamos dividir por 0
        return np.zeros(im.shape, np.uint8)
    return np.clip(im / tope * 255, 0, 255).astype(np.uint8)


def mostrar(imagenes, filas=1, archivo=None, tam=3.2, carpeta_res=None):
    '''Muestra un diccionario {"título": imagen} en una grilla y opcionalmente
    guarda la figura en results/modulo1/<archivo>.'''
    n = len(imagenes)
    cols = int(np.ceil(n / filas))              # columnas necesarias (redondeo hacia arriba)
    fig, ax = plt.subplots(filas, cols, figsize=(tam * 1.4 * cols, tam * filas), squeeze=False)
    ax = ax.ravel()                             # grilla 2D -> lista 1D, más fácil de recorrer
    for a, (titulo, im) in zip(ax, imagenes.items()):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.set_title(titulo, fontsize=10)
    for a in ax:
        a.axis("off")                           # sin ejes (también apaga los casilleros vacíos)
    plt.tight_layout()
    if archivo:
        if carpeta_res is None:
            raise ValueError("Indicá carpeta_res para guardar la figura.")
        fig.savefig(carpeta_res / archivo, dpi=150)
    plt.show()


def zoom(im, limites):
    '''Recorta la zona de la pupila de cualquier imagen del mismo tamaño que la ROI.'''
    zy0, zy1, zx0, zx1 = limites
    return im[zy0:zy1, zx0:zx1]


def gradiente(im, kx, ky):
    '''Magnitud del gradiente con un par de kernels (kx detecta cambios en x,
    ky en y).

    cv2.filter2D hace la "convolución" (técnicamente correlación) del kernel
    con la imagen. ddepth=cv2.CV_64F pide el resultado en float64: las
    derivadas pueden ser NEGATIVAS (de claro a oscuro) y en uint8 se perderían.
    '''
    im = im.astype(np.float64)
    gx = cv2.filter2D(im, cv2.CV_64F, kx)
    gy = cv2.filter2D(im, cv2.CV_64F, ky)
    return np.sqrt(gx ** 2 + gy ** 2)          # magnitud: combina ambas direcciones


# Kernels de Roberts y Prewitt, utilizados por detectar_bordes.
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
prewitt_y = prewitt_x.T


def detectar_bordes(im):
    '''Aplica los 5 operadores a una imagen y devuelve un diccionario
    {nombre: imagen de bordes lista para mostrar (uint8)}.'''
    res = {}
    res["Roberts"] = a_0_255(gradiente(im, roberts_x, roberts_y))
    res["Prewitt"] = a_0_255(gradiente(im, prewitt_x, prewitt_y))

    # Sobel ya viene en OpenCV: (1, 0) = derivada en x; (0, 1) = derivada en y.
    sx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
    res["Sobel"] = a_0_255(np.sqrt(sx ** 2 + sy ** 2))

    # Laplaciano: segunda derivada. Vale ~0 en zonas planas y en rampas
    # uniformes; cambia de signo justo SOBRE el borde. Mostramos su valor absoluto.
    res["Laplaciano"] = a_0_255(cv2.Laplacian(im, cv2.CV_64F, ksize=1))

    # Canny necesita dos umbrales: un píxel con gradiente > alto es borde
    # seguro; uno entre bajo y alto es borde solo si está CONECTADO a un borde
    # seguro (histéresis). Así sigue contornos sin agregar puntos sueltos.
    # Regla práctica para que funcione con cualquier foto: umbrales
    # proporcionales a la mediana de la imagen (±33 %).
    med = np.median(im)
    bajo, alto = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    res[f"Canny ({bajo}, {alto})"] = cv2.Canny(im, bajo, alto)   # ya da 0 o 255
    return res


def reconstruir(mag, fas):
    '''Arma F = mag · e^(j·fas) y aplica la TFD inversa. Devuelve la parte real
    (la parte imaginaria es ~0 salvo errores numéricos de redondeo).'''
    return np.real(np.fft.ifft2(mag * np.exp(1j * fas)))   # 1j = unidad imaginaria en Python


def psd(im):
    '''PSD centrada: |TFD|² con la frecuencia cero en el medio.'''
    return np.abs(np.fft.fftshift(np.fft.fft2(im.astype(np.float64)))) ** 2

def perfil_radial(P):
    '''Promedio de P sobre anillos concéntricos alrededor del centro.

    Frecuencia NORMALIZADA: dividimos la distancia al centro por la mitad del
    tamaño en cada eje, así 0 = continua y 1 = frecuencia máxima (Nyquist) en
    ambos ejes aunque la ROI no sea cuadrada.'''
    Nf, Nc = P.shape
    v, u = np.mgrid[0:Nf, 0:Nc]                    # coordenadas de fila (v) y columna (u)
    rad = np.hypot((u - Nc // 2) / (Nc / 2), (v - Nf // 2) / (Nf / 2))  # distancia normalizada
    bins = np.linspace(0, 1, 51)                    # 50 anillos entre 0 y 1
    idx = np.digitize(rad.ravel(), bins)            # a qué anillo pertenece cada punto
    # Promedio de P en cada anillo (np.bincount suma por anillo; dividimos por la cantidad)
    suma = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)
    cant = np.bincount(idx, minlength=len(bins) + 1)
    perfil = suma[1:len(bins)] / np.maximum(cant[1:len(bins)], 1)
    centros = (bins[:-1] + bins[1:]) / 2            # frecuencia del centro de cada anillo
    return centros, perfil


def filtrar_frecuencia(im, H):
    '''TFD -> centrar -> multiplicar por la máscara H -> descentrar -> TFD inversa.'''
    Fc = np.fft.fftshift(np.fft.fft2(im.astype(np.float64)))
    return np.real(np.fft.ifft2(np.fft.ifftshift(Fc * H)))


def a_uint8(im):
    '''Recorta a 0..255 y pasa a uint8 (para medir con las mismas reglas que una imagen real).'''
    return np.clip(np.round(im), 0, 255).astype(np.uint8)
