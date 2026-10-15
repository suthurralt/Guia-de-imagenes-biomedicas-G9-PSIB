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
    carga_muestreo.py           <- 1.1: carga, ROI, submuestreo, cuantización, color
    calidad_filtrado.py         <- 1.2: histograma, ecualización, escalado, filtrado espacial
    bordes_frecuencia.py        <- 1.3: bordes, TF 2D, PSD, filtrado frecuencial
  modulo2/
    segmentacion.py           <- 2.1: Otsu, K-means, Watershed, Snake
    morfologia.py             <- 2.2: dilatación, erosión, apertura, cierre, esqueletización
    anotacion.py              <- 2.2: anotación manual --> máscara de referencia
    metricas.py               <- 2.2: Recall, precisión, Dice, Jaccard
  modulo4/
    compresion.py             <- 4: Huffman, RLE (H/V), JPEG, PSNR/SSIM, blocking, diámetro
notebooks/
  01_1_carga_muestreo_color.ipynb
  01_2_calidad_intensidad_color.ipynb
  01_3_bordes_frecuencia_refactorizado.ipynb
  02_1_segmentacion.ipynb
  02_2_morfologia.ipynb
  04_compresion.ipynb
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
