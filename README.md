# Familias de estrellas: cúmulos abiertos en Gaia DR3 con CUDA

CC3069 Computación Paralela y Distribuida · Grupo 8 · Caso 8 (Ciencia espacial)
Luis Palacios, Hugo Barillas, Víctor Pérez

Buscamos cúmulos de estrellas (grupos que nacieron juntos) en el disco de la
galaxia agrupando ~38 millones de estrellas de Gaia DR3 por posición, paralaje y
movimiento propio (5 dimensiones) con DBSCAN. La misma tarea, con los mismos
datos y parámetros, se resuelve en CPU (scikit-learn) y en GPU (kernels CUDA
propios escritos con CuPy). Los resultados se resumen en [`RESULTADOS.md`](RESULTADOS.md).

## Cómo correrlo

Requisitos: Python 3.11+, GPU NVIDIA con driver CUDA 12/13 (probado en una RTX 3050
Laptop de 4 GB, Windows 11). Para CUDA 12 cambie `cupy-cuda13x` por `cupy-cuda12x`.

```bash
uv venv .venv && uv pip install -r requirements.txt     # o: python -m venv .venv && pip install -r requirements.txt
cd src
python gaia_download.py count      # COUNT(*) por región antes de descargar (~2 min)
python gaia_download.py download   # ~38.6 M estrellas, ~3.25 GB en FITS (~20 min con 8 conexiones)
python prepare.py                  # limpieza, teselas, estandarización, eps
python benchmark.py                # CPU vs GPU en 1x, 3x y 10x (~8 min, casi todo es la CPU)
python plots.py                    # figuras en figures/
```

El catálogo de referencia de cúmulos conocidos (Cantat-Gaudin et al. 2020) se
descarga de VizieR:

```bash
mkdir -p data/catalogs
curl -L "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/A%2BA/640/A1/table1&-out.max=unlimited&-out=Cluster,RA_ICRS,DE_ICRS,GLON,GLAT,r50,nbstars07,pmRA*,pmDE,plx" -o data/catalogs/cantat_gaudin_2020.tsv
```

`python src/dbscan_gpu.py` corre una prueba rápida con datos sintéticos que
compara el kernel contra scikit-learn (debe dar ARI = 1.0).

## Estructura

| Archivo | Qué hace |
|---|---|
| `src/config.py` | Todos los parámetros: filtros ADQL, teselas, `min_samples`, percentil de eps, escalas |
| `src/gaia_download.py` | Consultas ADQL al TAP de la ESA por píxeles HEALPix (rangos de `source_id`) |
| `src/prepare.py` | Lectura, limpieza, teselas de 12°, z-score por tesela (CPU y GPU), eps por tesela |
| `src/dbscan_cpu.py` | Línea base: `sklearn.cluster.DBSCAN` (KD-tree, `n_jobs=-1`) por tesela |
| `src/dbscan_gpu.py` | DBSCAN en GPU: rejilla 5D + radix sort + 4 kernels CUDA (núcleos, union-find atómico, bordes) |
| `src/benchmark.py` | T_CPU, T_GPU por etapa (eventos CUDA), transferencias, speedup, ARI, escalas 1x/3x/10x |
| `src/validate.py` | Resumen de cada grupo y cruce con Cantat-Gaudin et al. (2020) |
| `src/plots.py` | Figuras del informe |
| `results/` | `benchmark.json`, `prepare.json`, `clusters.csv`, `chunk_counts.csv`, bitácora de descarga |
| `figures/` | Gráficas para el informe y la presentación |

## Datos

* **Fuente:** Gaia DR3, tabla `gaiadr3.gaia_source`, archivo de la ESA
  (<https://gea.esac.esa.int/archive/>), vía TAP/ADQL.
* **Selección:** disco galáctico |b| < 10°, `parallax_over_error > 10`, `ruwe < 1.4`,
  G < 18, con movimiento propio. Columnas: `source_id, ra, dec, l, b, parallax,
  parallax_error, pmra, pmdec, phot_g_mean_mag, bp_rp, radial_velocity, ruwe`.
* **Descarga:** el servicio anónimo corta cada consulta a los 60 s y por defecto
  devuelve 50 000 filas, así que se pide por píxeles HEALPix nivel 4 (rangos
  contiguos e indexados de `source_id`, 992 bloques), en FITS binario y con
  `MAXREC` alto; si un bloque no termina a tiempo se divide en 4 automáticamente.
* **Licencia:** los datos de Gaia se distribuyen bajo CC BY-SA 3.0 IGO. Cita
  requerida: Gaia Collaboration, Vallenari et al. (2023), A&A 674, A1.
* **Catálogo de validación:** Cantat-Gaudin et al. (2020), A&A 640, A1
  (VizieR J/A+A/640/A1).
