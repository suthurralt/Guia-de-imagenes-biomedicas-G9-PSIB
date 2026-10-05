# Análisis del reflejo fotomotor pupilar — PSIB

Pipeline de procesamiento de imágenes biomédicas sobre videos de la pupila
(Procesamiento de Señales e Imágenes Biomédicas, ITBA).

## Estructura

```
config.yaml            <- ÚNICO archivo a editar para cambiar imagen/video/ROI
requirements.txt       <- dependencias
src/
  config.py            <- lectura del config
  modulo1/
    carga_muestreo.py  <- 1.1: carga, ROI, submuestreo, cuantización, color
notebooks/
  01_1_carga_muestreo_color.ipynb
data/
  frames/              <- frames extraídos (sí se suben)
  videos/              <- videos (NO se suben, ver .gitignore)
results/               <- figuras generadas por los notebooks
```

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Poner la imagen en `data/frames/` (o el video en `data/videos/`).
2. Editar `config.yaml`: `imagen_path` (o `imagen_path: null` + `video_path`) y la `roi`.
3. Abrir el notebook y ejecutar *Run All*.
