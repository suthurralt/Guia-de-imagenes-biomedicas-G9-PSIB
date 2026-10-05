"""
src/config.py — Lectura centralizada del archivo config.yaml.

¿Por qué un módulo aparte solo para esto?
    Porque todos los módulos del TP (1, 2, 3, 4) van a necesitar leer la
    configuración. Si cada uno la leyera a su manera, cambiar algo implicaría
    tocar varios archivos. Acá se resuelve una sola vez.
"""

# Permite escribir anotaciones de tipo como "str | Path" aunque se use
# Python 3.8/3.9 (la guía pide v3.8+; esa sintaxis nativa es de 3.10).
from __future__ import annotations

# pathlib es la librería estándar de Python para manejar rutas de archivos.
# Path("a") / "b" arma "a/b" de forma correcta en Linux, Mac y Windows.
from pathlib import Path

# PyYAML: convierte el texto del .yaml en diccionarios/listas de Python.
import yaml

# __file__ es la ruta de ESTE archivo (src/config.py).
# .resolve() la vuelve absoluta; .parent sube una carpeta (-> src/);
# otro .parent sube otra (-> raíz del repo).
# Así la raíz del repo se encuentra sola, sin importar desde dónde
# se ejecute el código (terminal, notebook, VS Code...).
RAIZ_REPO = Path(__file__).resolve().parent.parent


def cargar_config(ruta: str | Path | None = None) -> dict:
    """Lee config.yaml y devuelve su contenido como diccionario.

    Parámetros
    ----------
    ruta : ruta al .yaml. Si es None, usa <raíz del repo>/config.yaml.

    Devuelve
    --------
    dict con la misma estructura que el YAML. Ej.: cfg["roi"]["x0"].
    """
    # Si no nos pasan ruta, usamos la por defecto.
    if ruta is None:
        ruta = RAIZ_REPO / "config.yaml"

    # "with open(...)" abre el archivo y lo cierra solo al terminar el bloque,
    # aunque ocurra un error. encoding="utf-8" para que los acentos no rompan.
    with open(ruta, "r", encoding="utf-8") as f:
        # safe_load solo interpreta tipos básicos (números, textos, listas,
        # dicts). Es la forma segura: nunca ejecuta código del archivo.
        return yaml.safe_load(f)


def ruta_absoluta(ruta_relativa: str | Path) -> Path:
    """Convierte una ruta escrita en el config (relativa a la raíz del repo)
    en una ruta absoluta. Si ya es absoluta, la deja como está."""
    p = Path(ruta_relativa)
    # is_absolute() -> True si empieza con "/" (Linux/Mac) o "C:\" (Windows).
    return p if p.is_absolute() else RAIZ_REPO / p
