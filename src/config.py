"""Parametros compartidos por todo el pipeline (descarga, limpieza, CPU y GPU)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# --- Seleccion en el archivo de Gaia (ADQL) ---------------------------------
TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
COLUMNS = [
    "source_id", "ra", "dec", "l", "b", "parallax", "parallax_error",
    "pmra", "pmdec", "phot_g_mean_mag", "bp_rp", "radial_velocity", "ruwe",
]
B_MAX = 10.0            # |b| < B_MAX grados: disco galactico
PARALLAX_OVER_ERROR = 10.0
RUWE_MAX = 1.4
G_MAX = 18.0

# Descarga por bloques de source_id. source_id codifica el pixel HEALPix
# nivel 12 (anidado) en sus bits altos: hpx12 = source_id // 2**35.
# Un pixel HEALPix nivel 2 (192 en todo el cielo) = 2**20 pixeles nivel 12,
# es decir, un rango contiguo de source_id de tamano 2**55. Ese rango esta
# indexado en el archivo, por eso estas consultas son rapidas.
HPX_LEVEL = 2
N_CHUNKS = 12 * 4 ** HPX_LEVEL
CHUNK_SPAN = 2 ** 35 * 4 ** (12 - HPX_LEVEL)

# --- Clustering ---------------------------------------------------------------
FEATURES = ["l", "b", "parallax", "pmra", "pmdec"]   # espacio 5D
TILE_DEG = 12.0          # teselas de 12x12 grados (como Castro-Ginard et al. 2020)
EPS_PERCENTILE = 0.2    # eps por tesela = percentil 0.2 de la distancia al k-esimo vecino
MIN_SAMPLES = 10         # estrellas minimas para un nucleo DBSCAN

# Nivel HEALPix de los bloques de descarga (16 bloques por pixel nivel 2).
DOWNLOAD_LEVEL = 4

# Filtro de candidatos: grupos con muy pocas estrellas suelen ser fluctuaciones
# del campo, y grupos enormes suelen ser regiones densas del disco, no cumulos.
CANDIDATE_MIN = 20
CANDIDATE_MAX = 2000

# Escalas para medir como crece el tiempo (1x, 3x, 10x de datos): teselas usadas.
SCALES = {
    "1x": [0, 10, 20],
    "3x": [0, 1, 2, 10, 11, 12, 20, 21, 22],
    "10x": list(range(30)),
}
GPU_MAX_POINTS = 14_000_000   # estrellas por lote en la GPU (RTX 3050 de 4 GB)
