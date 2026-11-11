# Fase 2: resultados (Grupo 8, Familias de estrellas)

Todo lo de este documento se midió el 11 de noviembre de 2026 en una laptop con
**GPU NVIDIA GeForce RTX 3050 Laptop (4 GB, 16 SM, CUDA 13.1)** y **CPU Intel de 12
hilos (familia Alder Lake), 16 GB de RAM**, Windows 11. Los números salen de
`results/benchmark.json` y `results/prepare.json`, y se regeneran con `python src/benchmark.py`.

## 1. Datos

| | |
|---|---|
| Fuente | Gaia DR3, `gaiadr3.gaia_source`, archivo de la ESA, <https://gea.esac.esa.int/archive/> |
| Consulta | disco galáctico \|b\| < 10°, `parallax_over_error > 10`, `ruwe < 1.4`, G < 18, con `pmra`/`pmdec` |
| COUNT antes de bajar | 38 601 643 estrellas (`results/chunk_counts.csv`) |
| Descargado | **38 601 643 filas, 3.25 GB** en 992 archivos FITS (19 min, 0 errores) |
| Limpio | 38 601 643 filas (0 duplicados, 0 nulos en las 5 variables), parquet de 3.3 GB |
| Licencia | CC BY-SA 3.0 IGO · Gaia Collaboration, Vallenari et al. (2023), A&A 674, A1 |
| Validación | Cantat-Gaudin et al. (2020), A&A 640, A1: 2017 cúmulos abiertos (VizieR J/A+A/640/A1) |

**Cambio respecto a la propuesta:** la propuesta decía "por franjas de ascensión
recta". En la práctica, el servicio anónimo corta cada consulta a los **60 s** y por
defecto devuelve solo **50 000 filas**. Filtrar por `l`/`b` recorre toda la tabla
y no termina a tiempo. La solución fue pedir por **píxeles HEALPix** usando rangos
de `source_id` (Gaia codifica el píxel HEALPix nivel 12 en los bits altos de
`source_id`, y esa columna está indexada), con `MAXREC` alto y formato FITS binario,
que es ~2× más rápido que CSV.

**Preparación:** el disco se divide en 30 teselas de 12° de longitud (como Castro-Ginard
et al. 2020). En cada tesela las 5 variables (l, b, paralaje, pmra, pmdec) se
estandarizan con z-score. `eps` se elige por tesela: el percentil 0.2 de la distancia
al 10.º vecino. Así se adapta a la densidad, porque el centro galáctico es mucho más
denso que el anticentro. `min_samples = 10`.

## 2. Solución: la misma tarea dos veces

* **CPU:** `sklearn.cluster.DBSCAN` (KD-tree, `n_jobs=-1` con los 12 hilos, float64),
  una tesela a la vez. Es lo que usaría un astrónomo en su computadora.
* **GPU:** DBSCAN propio con **kernels CUDA escritos a mano** (CuPy `RawModule`, `src/dbscan_gpu.py`):
  1. `compute_keys`: cada estrella cae en una celda de una rejilla 5D de lado eps.
     La llave de 64 bits junta la tesela y las 5 coordenadas de la celda.
  2. Radix sort por llave: las estrellas de cada celda quedan contiguas en memoria.
  3. `count_neighbors`: un hilo por estrella revisa las 3⁵ = 243 celdas vecinas
     (búsqueda binaria) y decide si es núcleo. Deja de contar al llegar a `min_samples`.
  4. `union_cores`: union-find sin bloqueos con `atomicCAS`, que une núcleos vecinos.
  5. `flatten` + `assign_border`: raíz de cada núcleo; los puntos borde toman la
     etiqueta del núcleo vecino.
  Las 30 teselas van en la misma llamada (la tesela está en los bits altos de la
  llave), así que la GPU recibe millones de hilos de una vez. Como solo tiene 4 GB,
  el disco completo se procesa en 3 lotes.

**¿Encuentran lo mismo?** Sí. El **Adjusted Rand Index entre CPU y GPU es 1.00000** en
las tres escalas, con el mismo número de grupos y 100 % de acuerdo en qué estrellas
son ruido. La GPU no solo es más rápida: da la misma respuesta.

## 3. Mediciones

T_CPU es una corrida (7 min en total); T_GPU es la mediana de 5 corridas, después de
un calentamiento que compila los kernels. Las escalas usan 3, 9 y 30 teselas
repartidas por el disco.

| Escala | Estrellas | T_CPU | T_GPU | Transferencia | % transf. | **Speedup** | Speedup solo kernels | Grupos | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | 3 726 142 | 40.7 s | 0.67 s | 0.028 s | 4 % | **61×** | 101× | 642 | 1.0 |
| 3x | 9 943 709 | 98.8 s | 1.92 s | 0.099 s | 5 % | **51×** | 89× | 1 339 | 1.0 |
| 10x | 38 601 643 | 412.9 s | 8.34 s | 0.380 s | 5 % | **50×** | 93× | 7 267 | 1.0 |

Desglose de la GPU en 10x (`figures/gpu_stages.png`), medido con eventos CUDA:

| Etapa | Tiempo | % |
|---|---:|---:|
| CPU: armar los 3 lotes (indexar 38 M filas) | 3.50 s | 42 % |
| Copia CPU→GPU | 0.28 s | 3 % |
| Rejilla + radix sort | 0.16 s | 2 % |
| Contar vecinos (núcleos) | 2.06 s | 25 % |
| Union-find | 0.26 s | 3 % |
| Bordes + etiquetas | 1.97 s | 24 % |
| Copia GPU→CPU | 0.10 s | 1 % |

Estandarización (etapa 1) con las 38.6 M estrellas: pandas 22.3 s; CuPy 2.74 s, de los
cuales 1.80 s (66 %) son copias. El resultado es idéntico (diferencia máxima 1.2e-7).

## 4. Análisis

**Cuello de botella.** En CPU, DBSCAN es el 71 % de todo el pipeline (413 de 582 s).
La búsqueda de vecinos en 5D es el trabajo dominante: un KD-tree pierde eficiencia
en 5 dimensiones y la expansión de grupos de scikit-learn corre en un solo hilo. En
GPU el kernel solo tarda 4.5 s (93× más rápido). Lo que queda es: (a) mover los
datos por PCIe, que es poco (5 %), y (b) **trabajo de CPU para partir los datos en
lotes que quepan en 4 GB** (42 % del tiempo de GPU). Por eso el speedup baja de 61×
a 50× al pasar a 10x: con 1x todo cabe en un solo lote. En una GPU de 16 GB (T4 en
Colab) sería un solo lote y el speedup debería acercarse a ~90×.

**Ley de Amdahl** (`figures/amdahl.png`). Pipeline completo con las 38.6 M estrellas:

| Etapa | CPU | GPU |
|---|---:|---:|
| Leer 992 FITS | 58.4 s | 58.4 s |
| Limpiar | 17.2 s | 17.2 s |
| Elegir eps (KD-tree sobre muestras) | 66.5 s | 66.5 s |
| Estandarizar | 22.3 s | 2.7 s |
| DBSCAN | 412.9 s | 8.3 s |
| Cruce con catálogo | 4.3 s | 4.3 s |
| **Total** | **581.5 s** | **157.4 s** |

La fracción que aceleramos es p = 0.748, así que el máximo teórico es
1/(1−p) = **4.0×**, aunque la GPU fuera infinitamente rápida. Obtuvimos **3.7×**
de punta a punta. Ahora el 93 % del tiempo de GPU está en lo que no paralelizamos:
leer archivos y elegir eps. El siguiente paso obvio sería mover la elección de eps
a la GPU (es otra búsqueda de vecinos) y leer en formato columnar directo a la GPU
(cuDF/Parquet). Además, la descarga del archivo (19 min) supera a todo el cómputo,
pero se hace una sola vez.

**¿Qué pasa con 10 veces más datos?** De 1x a 10x hay 10.4× más estrellas. T_CPU
creció 10.1× y T_GPU 12.5×; los dos escalan casi lineal porque las teselas son
independientes. Al extrapolar a 10 veces el disco actual (~386 M estrellas, p. ej.
bajando a G < 19 o quitando el corte en b), una corrida tardaría **~70 min en CPU
frente a ~1.5 min en GPU**. La memoria no es un límite, porque el código ya procesa
por lotes.

## 5. Resultado e impacto

* **2 911 grupos candidatos** (20 a 2 000 estrellas) en el disco galáctico (`results/clusters.csv`, `figures/sky_map.png`).
* **1 159 de los 1 837 cúmulos conocidos** de Cantat-Gaudin+2020 en la región se
  recuperan (63 %) sin haberle dicho nada al algoritmo sobre ellos. Ejemplo: **NGC 7789**,
  1 717 estrellas con la secuencia principal, el "red clump" y la rama de gigantes
  claramente separados del campo (`figures/example_cluster.png`).
* **1 656 grupos no coinciden con el catálogo.** Son *candidatos*, no descubrimientos:
  muchos pueden ser fluctuaciones del campo (se concentran en algunas teselas, como
  l ≈ 50–75°). Ahí entra el astrónomo: revisar su diagrama color-magnitud antes de
  pedir tiempo de telescopio.

**Qué cambia para el observatorio.** Una búsqueda de 7 minutos baja a 8 segundos.
Lo importante es que eso permite **explorar parámetros**: probar 50 combinaciones de
`eps`/`min_samples` toma ~6 horas en CPU y ~7 minutos en GPU, en una laptop de
estudiante. Así el equipo puede ver qué cúmulos son robustos ante cualquier
parámetro (los que aparecen siempre) y llegar a la convocatoria con una lista
justificada en lugar de una corrida única. Con 10 veces más datos, ese mismo barrido
sería de ~2.5 días en CPU frente a ~1.2 horas en GPU: la diferencia entre llegar o
no al plazo de la convocatoria.

## 6. Limitaciones (para decirlas antes de que nos las pregunten)

* Las teselas no se traslapan: un cúmulo justo en un borde de 12° se parte en dos.
* `eps` por percentil es un criterio simple; Castro-Ginard et al. usan una prueba
  estadística contra un campo aleatorio y una red neuronal sobre el diagrama color-magnitud.
* El criterio de coincidencia con el catálogo (separación < max(2·r50, 0.5°),
  Δparalaje < 0.3 mas, Δmov. propio < 0.7 mas/año) es aproximado.
* La CPU usa float64 y la GPU float32. No cambió ninguna etiqueta (ARI = 1.0), pero
  en otros datos podría mover alguna estrella justo en el borde de eps.
* T_CPU es de una sola corrida porque tarda 7 minutos; la GPU es mediana de 5.
* No usamos cuML (RAPIDS no corre nativo en Windows). Escribimos los kernels,
  lo que además deja ver qué hace cada etapa.


## Referencias

* Gaia Collaboration, Vallenari, A., et al. (2023). *Gaia Data Release 3: Summary of the content and survey properties.* A&A 674, A1.
* Cantat-Gaudin, T., et al. (2020). *Painting a portrait of the Galactic disc with its stellar clusters.* A&A 640, A1.
* Castro-Ginard, A., et al. (2020). *Hunting for open clusters in Gaia DR2: 582 new open clusters in the Galactic disc.* A&A 635, A45.
* Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD-96.
* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. · CuPy: Okuta, R., et al. (2017), LearningSys.
