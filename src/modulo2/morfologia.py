
"""
src/modulo2/morfologia.py

Operaciones morfológicas sobre la máscara binaria de la pupila.

Incluye:
- Erosión
- Dilatación
- Apertura
- Cierre
- Combinaciones de operaciones
- Esqueletización del contorno
- Comparación de resultados
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ============================================================
# 1. FUNCIONES AUXILIARES
# ============================================================

def _preparar_mascara(mascara):
    """
    Convierte una máscara binaria a uint8:
    pupila = 255 (blanco)
    fondo = 0 (negro)

    IMPORTANTE:
    La entrada debe identificar previamente la pupila.
    No pasar directamente etiquetas de Watershed sin
    seleccionar antes la etiqueta correspondiente.
    """
    return np.where(mascara > 0, 255, 0).astype(np.uint8)


def _crear_kernel(k=3, kernel_forma="ellipse"):
    """
    Crea el elemento estructurante.

    Parámetros:
        k: tamaño impar del kernel.
        kernel_forma: 'ellipse', 'rect' o 'cross'.
    """
    if not isinstance(k, (int, np.integer)) or k < 1 or k % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser un entero positivo impar.")

    # Pueden seleccionarse otras formas de kernel, en este caso se utiliza la elipse por la forma de la pupila.
    formas = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }

    if kernel_forma not in formas:
        raise ValueError(
            "kernel_forma debe ser 'ellipse', 'rect' o 'cross'."
        )

    return cv2.getStructuringElement(
        formas[kernel_forma], (k, k)
    )


def _validar_iteraciones(iteraciones):
    """Valida el número de iteraciones morfológicas."""
    if not isinstance(iteraciones, (int, np.integer)) or iteraciones < 1:
        raise ValueError("iteraciones debe ser un entero mayor o igual a 1.")


# ============================================================
# 2. OPERACIONES MORFOLÓGICAS INDIVIDUALES
# ============================================================

# k, iteraciones y kernel_forma son valores puestos para nuestro video en particular, y se encuentran almacenados en config.yaml
# Para modificarlos, se deben realizar cambios en ese archivo.

def erosion(mascara, k, iteraciones, kernel_forma):
    """
    Reduce las regiones blancas de la máscara.

    Puede eliminar pequeñas regiones blancas y reducir salientes,
    pero también achica la pupila.
    """
    mascara = _preparar_mascara(mascara)
    kernel = _crear_kernel(k, kernel_forma)
    _validar_iteraciones(iteraciones)

    return cv2.erode(
        mascara, kernel, iterations=iteraciones
    )


def dilatacion(mascara, k, iteraciones, kernel_forma):
    """
    Expande las regiones blancas.

    Puede conectar pequeñas discontinuidades, pero también
    agranda el área segmentada de la pupila.
    """
    mascara = _preparar_mascara(mascara)
    kernel = _crear_kernel(k, kernel_forma)
    _validar_iteraciones(iteraciones)

    return cv2.dilate(
        mascara, kernel, iterations=iteraciones
    )


def apertura(mascara, k, iteraciones, kernel_forma):
    """
    Erosión seguida de dilatación.

    Puede eliminar pequeños componentes blancos espurios.
    """
    mascara = _preparar_mascara(mascara)
    kernel = _crear_kernel(k, kernel_forma)
    _validar_iteraciones(iteraciones)

    return cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iteraciones
    )


def cierre(mascara, k, iteraciones, kernel_forma):
    """
    Dilatación seguida de erosión.

    Puede cerrar pequeños huecos negros y discontinuidades.
    """
    mascara = _preparar_mascara(mascara)
    kernel = _crear_kernel(k, kernel_forma)
    _validar_iteraciones(iteraciones)

    return cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iteraciones
    )


# ============================================================
# 3. COMBINACIONES MORFOLÓGICAS
# ============================================================

def apertura_cierre(mascara, k, iteraciones, kernel_forma):
    """
    Aplica apertura y luego cierre.

    Primero intenta eliminar pequeñas regiones blancas espurias
    y después cerrar pequeños huecos.
    """
    resultado = apertura(
        mascara, k, iteraciones, kernel_forma
    )

    return cierre(
        resultado, k, iteraciones, kernel_forma
    )


def cierre_apertura(mascara, k, iteraciones, kernel_forma):
    """
    Aplica cierre y luego apertura.

    Primero intenta cerrar pequeños huecos y después eliminar
    pequeñas regiones blancas espurias.
    """
    resultado = cierre(
        mascara, k, iteraciones, kernel_forma
    )

    return apertura(
        resultado, k, iteraciones, kernel_forma
    )


# ============================================================
# 4. ESQUELETIZACIÓN DEL CONTORNO PUPILAR
# ============================================================

def esqueletizar_contorno(mascara, k, kernel_forma):
    """
    Obtiene el contorno de la pupila mediante el gradiente
    morfológico y luego lo adelgaza mediante esqueletización.

    Devuelve una imagen uint8:
    - 255: píxeles del contorno esqueletizado.
    - 0: fondo.

    El resultado representa el borde, no el esqueleto del
    interior de la región pupilar.
    
    """
    mascara = _preparar_mascara(mascara)

    kernel = _crear_kernel(k, kernel_forma)

    borde = cv2.morphologyEx(
        mascara, cv2.MORPH_GRADIENT, kernel
    )

    esqueleto = skeletonize(borde > 0)

    return esqueleto.astype(np.uint8) * 255

# ============================================================
# 5. COMPARACIÓN DE OPERACIONES
# ============================================================

def comparar_morfologia(
    mascara,
    k,
    iteraciones,
    kernel_forma
):
    """
    Aplica las operaciones individuales y las combinaciones.

    Devuelve un diccionario:
        nombre de operación -> máscara resultante

    La máscara original también se incluye como referencia.
    """
    mascara = _preparar_mascara(mascara)

    resultados = {
        "Original": mascara,
        "Erosion": erosion(
            mascara, k, iteraciones, kernel_forma
        ),
        "Dilatacion": dilatacion(
            mascara, k, iteraciones, kernel_forma
        ),
        "Apertura": apertura(
            mascara, k, iteraciones, kernel_forma
        ),
        "Cierre": cierre(
            mascara, k, iteraciones, kernel_forma
        ),
        "Apertura + cierre": apertura_cierre(
            mascara, k, iteraciones, kernel_forma
        ),
        "Cierre + apertura": cierre_apertura(
            mascara, k, iteraciones, kernel_forma
        ),
    }

    resultados["Esqueleto del contorno"] = esqueletizar_contorno(
        mascara, k, kernel_forma
    )

    return resultados
