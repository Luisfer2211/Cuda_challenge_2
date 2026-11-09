"""DBSCAN en GPU con kernels CUDA escritos a mano (CuPy RawModule).

Idea (variante de G-DBSCAN / CUDA-DClust con rejilla):
  1. Cada estrella cae en una celda de una rejilla 5D de lado eps. La llave de la
     celda junta (tesela, c_l, c_b, c_plx, c_pmra, c_pmdec) en un entero de 64 bits.
  2. Se ordenan las estrellas por llave (radix sort de CuPy). Asi las estrellas de
     una misma celda quedan contiguas en memoria.
  3. Kernel `count_neighbors`: un hilo por estrella. Revisa las 3^5 = 243 celdas
     vecinas (busqueda binaria de cada llave) y cuenta cuantas estrellas estan a
     distancia <= eps. Si llega a min_samples es "nucleo" (corta antes).
  4. Kernel `union_cores`: un hilo por nucleo. Une con union-find atomico
     (atomicCAS) a todos los nucleos vecinos. Cada componente conexa = un cumulo.
     Kernel `flatten`: cada nucleo guarda la raiz de su componente.
  5. Kernel `assign_border`: las estrellas no nucleo toman la etiqueta de algun
     nucleo vecino (borde) o quedan como ruido (-1).

Es exactamente la definicion de DBSCAN que usa scikit-learn (la estrella cuenta
como su propia vecina, distancia euclidiana <= eps). La unica diferencia posible
es en estrellas borde que tocan dos cumulos: DBSCAN no define a cual van.

Todas las teselas se procesan en la misma llamada: la tesela va en los bits
altos de la llave, asi que nunca se mezclan vecinos de teselas distintas y la
GPU recibe millones de hilos de una vez.
"""
from __future__ import annotations

import time

import cupy as cp
import numpy as np

BITS = 11                  # bits por dimension en la llave de celda
MAXC = (1 << BITS) - 1     # 2047
CENTER = 1 << (BITS - 1)   # 1024: desplaza las coordenadas para que sean >= 0

_SRC = r"""
#define D 5
#define BITS %(BITS)d
#define MAXC %(MAXC)d
#define CENTER %(CENTER)d
#define NOFF 243

__device__ __forceinline__ void decode(long long key, int* c, int* tile) {
    for (int d = D - 1; d >= 0; --d) { c[d] = (int)(key & MAXC); key >>= BITS; }
    *tile = (int)key;
}

// busqueda binaria de una llave de celda en la lista ordenada de celdas unicas
__device__ __forceinline__ int find_cell(const long long* __restrict__ ucell, int ncell, long long key) {
    int lo = 0, hi = ncell;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (ucell[mid] < key) lo = mid + 1; else hi = mid;
    }
    return (lo < ncell && ucell[lo] == key) ? lo : -1;
}

// llave de la celda vecina numero `off` (0..242); -1 si se sale de la rejilla
__device__ __forceinline__ long long neighbor_key(const int* c, int tile, int off) {
    long long key = tile;
    for (int d = 0; d < D; ++d) {
        int cc = c[d] + (off %% 3) - 1;
        off /= 3;
        if (cc < 0 || cc > MAXC) return -1;
        key = (key << BITS) | cc;
    }
    return key;
}

__device__ __forceinline__ float dist2(const float* __restrict__ X, int i, const float* xi) {
    float s = 0.f;
    for (int d = 0; d < D; ++d) { float t = X[i * D + d] - xi[d]; s += t * t; }
    return s;
}

extern "C" __global__ void compute_keys(const float* __restrict__ X, const int* __restrict__ tile,
                                        const float* __restrict__ inv_eps, int n,
                                        long long* __restrict__ keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = tile[i];
    float ie = inv_eps[t];
    long long key = t;
    for (int d = 0; d < D; ++d) {
        int c = (int)floorf(X[i * D + d] * ie) + CENTER;
        c = c < 0 ? 0 : (c > MAXC ? MAXC : c);   // recortar no rompe la correccion
        key = (key << BITS) | c;
    }
    keys[i] = key;
}

extern "C" __global__ void count_neighbors(const float* __restrict__ X, const long long* __restrict__ keys,
                                           const long long* __restrict__ ucell, const int* __restrict__ cstart,
                                           int ncell, const float* __restrict__ eps2, int n, int min_samples,
                                           unsigned char* __restrict__ is_core) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c[D], t;
    decode(keys[i], c, &t);
    float xi[D];
    for (int d = 0; d < D; ++d) xi[d] = X[i * D + d];
    float e2 = eps2[t];
    int cnt = 0;
    for (int off = 0; off < NOFF && cnt < min_samples; ++off) {
        long long k = neighbor_key(c, t, off);
        if (k < 0) continue;
        int cell = find_cell(ucell, ncell, k);
        if (cell < 0) continue;
        for (int j = cstart[cell]; j < cstart[cell + 1]; ++j) {
            if (dist2(X, j, xi) <= e2 && ++cnt >= min_samples) break;
        }
    }
    is_core[i] = cnt >= min_samples;
}

__device__ int find_root(volatile int* parent, int x) {
    while (true) {
        int p = parent[x];
        if (p == x) return x;
        int gp = parent[p];
        parent[x] = gp;          // path halving: gp siempre es ancestro de x
        x = gp;
    }
}

__device__ void unite(int* parent, int a, int b) {
    while (true) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a == b) return;
        if (a < b) { int tmp = a; a = b; b = tmp; }
        int old = atomicCAS(&parent[a], a, b);   // la raiz mayor cuelga de la menor
        if (old == a) return;
        a = old;
    }
}

extern "C" __global__ void union_cores(const float* __restrict__ X, const long long* __restrict__ keys,
                                       const long long* __restrict__ ucell, const int* __restrict__ cstart,
                                       int ncell, const float* __restrict__ eps2, int n,
                                       const unsigned char* __restrict__ is_core, int* parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !is_core[i]) return;
    int c[D], t;
    decode(keys[i], c, &t);
    float xi[D];
    for (int d = 0; d < D; ++d) xi[d] = X[i * D + d];
    float e2 = eps2[t];
    for (int off = 0; off < NOFF; ++off) {
        long long k = neighbor_key(c, t, off);
        if (k < 0) continue;
        int cell = find_cell(ucell, ncell, k);
        if (cell < 0) continue;
        for (int j = cstart[cell]; j < cstart[cell + 1]; ++j) {
            if (j > i && is_core[j] && dist2(X, j, xi) <= e2) unite(parent, i, j);
        }
    }
}

// Despues de union_cores `parent` ya no cambia: se busca la raiz solo leyendo
// (sin compresion) y se escribe en otro arreglo, asi no hay carreras.
extern "C" __global__ void flatten(const int* __restrict__ parent, const unsigned char* __restrict__ is_core,
                                   int n, int* __restrict__ croot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !is_core[i]) return;
    int x = i;
    while (parent[x] != x) x = parent[x];
    croot[i] = x;
}

extern "C" __global__ void assign_border(const float* __restrict__ X, const long long* __restrict__ keys,
                                         const long long* __restrict__ ucell, const int* __restrict__ cstart,
                                         int ncell, const float* __restrict__ eps2, int n,
                                         const unsigned char* __restrict__ is_core,
                                         const int* __restrict__ croot, int* __restrict__ root) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (is_core[i]) { root[i] = croot[i]; return; }
    int c[D], t;
    decode(keys[i], c, &t);
    float xi[D];
    for (int d = 0; d < D; ++d) xi[d] = X[i * D + d];
    float e2 = eps2[t];
    int best = -1;     // el nucleo vecino de menor indice: resultado determinista
    for (int off = 0; off < NOFF; ++off) {
        long long k = neighbor_key(c, t, off);
        if (k < 0) continue;
        int cell = find_cell(ucell, ncell, k);
        if (cell < 0) continue;
        for (int j = cstart[cell]; j < cstart[cell + 1]; ++j) {
            if (is_core[j] && (best < 0 || j < best) && dist2(X, j, xi) <= e2) best = j;
        }
    }
    root[i] = best < 0 ? -1 : croot[best];
}
""" % {"BITS": BITS, "MAXC": MAXC, "CENTER": CENTER}

_mod = None


def _kernels():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
    return _mod


def _launch(name, n, *args, block=256):
    _kernels().get_function(name)(((n + block - 1) // block,), (block,), args)


class Timer:
    """Acumula tiempos por etapa usando eventos CUDA (miden tiempo real en GPU)."""

    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.start = cp.cuda.Event()
                self.end = cp.cuda.Event()
                self.start.record()

            def __exit__(self, *exc):
                self.end.record()
                self.end.synchronize()
                ms = cp.cuda.get_elapsed_time(self.start, self.end)
                timer.times[name] = timer.times.get(name, 0.0) + ms / 1000.0

        return _Ctx()


def dbscan_gpu(X: np.ndarray, tile: np.ndarray, eps: np.ndarray, min_samples: int,
               timer: Timer | None = None) -> np.ndarray:
    """DBSCAN por teselas en GPU.

    X: (n, 5) float32 ya estandarizado. tile: (n,) int32 id de tesela.
    eps: (n_tiles,) radio por tesela. Devuelve etiquetas (n,) int32, -1 = ruido.
    Las etiquetas son globales (unicas entre teselas).
    """
    timer = timer or Timer()
    n = len(X)
    with timer("transfer_h2d"):
        dX = cp.asarray(np.ascontiguousarray(X, dtype=np.float32))
        dtile = cp.asarray(tile.astype(np.int32))
        deps = cp.asarray(eps.astype(np.float32))
    with timer("grid_sort"):
        inv_eps = (1.0 / deps).astype(cp.float32)
        eps2 = (deps * deps).astype(cp.float32)
        keys = cp.empty(n, dtype=cp.int64)
        _launch("compute_keys", n, dX, dtile, inv_eps, np.int32(n), keys)
        del dtile
        order = cp.argsort(keys)
        keys = keys[order]
        dX = dX[order]
        # celdas unicas y donde empieza cada una dentro del arreglo ordenado
        is_start = cp.empty(n, dtype=cp.bool_)
        is_start[0] = True
        cp.not_equal(keys[1:], keys[:-1], out=is_start[1:])
        cstart = cp.nonzero(is_start)[0].astype(cp.int32)
        del is_start
        ucell = keys[cstart]
        cstart = cp.concatenate([cstart, cp.asarray([n], dtype=cp.int32)])
        ncell = np.int32(len(ucell))
    with timer("core_points"):
        is_core = cp.zeros(n, dtype=cp.uint8)
        _launch("count_neighbors", n, dX, keys, ucell, cstart, ncell, eps2,
                np.int32(n), np.int32(min_samples), is_core)
    with timer("union_find"):
        parent = cp.arange(n, dtype=cp.int32)
        _launch("union_cores", n, dX, keys, ucell, cstart, ncell, eps2, np.int32(n), is_core, parent)
        croot = cp.empty(n, dtype=cp.int32)
        _launch("flatten", n, parent, is_core, np.int32(n), croot)
        del parent
    with timer("border_labels"):
        root = cp.empty(n, dtype=cp.int32)
        _launch("assign_border", n, dX, keys, ucell, cstart, ncell, eps2, np.int32(n),
                is_core, croot, root)
        del dX, keys, ucell, cstart, croot, is_core
        # renumerar raices -> 0..K-1 y volver al orden original
        valid = root >= 0
        _, compact = cp.unique(root[valid], return_inverse=True)
        lab_sorted = cp.full(n, -1, dtype=cp.int32)
        lab_sorted[valid] = compact.astype(cp.int32).ravel()
        labels = cp.empty(n, dtype=cp.int32)
        labels[order] = lab_sorted
        del root, valid, compact, lab_sorted, order
    with timer("transfer_d2h"):
        out = cp.asnumpy(labels)
    del labels
    cp.get_default_memory_pool().free_all_blocks()
    return out


def dbscan_gpu_batched(X, tile, eps, min_samples, max_points, timer=None):
    """Igual que dbscan_gpu, pero agrupa teselas en lotes de <= max_points
    estrellas para no pasarse de la memoria de la GPU (4 GB en la RTX 3050)."""
    timer = timer or Timer()
    labels = np.full(len(X), -1, dtype=np.int32)
    counts = np.bincount(tile, minlength=len(eps))
    batches, cur, size = [], [], 0
    for t in np.flatnonzero(counts):
        if cur and size + counts[t] > max_points:
            batches.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += counts[t]
    if cur:
        batches.append(cur)
    offset = 0
    for b in batches:
        idx = np.flatnonzero(np.isin(tile, b))
        lab = dbscan_gpu(X[idx], tile[idx], eps, min_samples, timer)
        labels[idx] = np.where(lab >= 0, lab + offset, -1)
        offset += int(lab.max()) + 1
    return labels, len(batches)


def warmup():
    """Compila los kernels (NVRTC) para que la compilacion no cuente en las mediciones."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    dbscan_gpu(X, np.zeros(2000, np.int32), np.array([0.5], np.float32), 5)
    cp.cuda.Device().synchronize()


if __name__ == "__main__":
    # prueba rapida contra scikit-learn con datos sinteticos
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    warmup()
    for n in (20_000, 200_000):
        X, _ = make_blobs(n_samples=n, centers=40, n_features=5, cluster_std=0.4,
                          center_box=(-8, 8), random_state=1)
        X = np.vstack([X, np.random.default_rng(2).uniform(-10, 10, size=(n // 2, 5))]).astype(np.float32)
        tile = (X[:, 0] > 0).astype(np.int32)
        eps = np.array([0.35, 0.35], np.float32)
        t0 = time.perf_counter()
        ref = np.full(len(X), -1)
        off = 0
        for t in (0, 1):
            m = tile == t
            lab = DBSCAN(eps=0.35, min_samples=10, n_jobs=-1).fit_predict(X[m].astype(np.float64))
            ref[m] = np.where(lab >= 0, lab + off, -1)
            off += lab.max() + 1
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        got = dbscan_gpu(X, tile, eps, 10)
        tg = time.perf_counter() - t0
        print(f"n={len(X):,} CPU {tc:.2f}s GPU {tg:.3f}s ARI={adjusted_rand_score(ref, got):.5f} "
              f"clusters cpu={ref.max()+1} gpu={got.max()+1} ruido cpu={np.mean(ref<0):.3f} gpu={np.mean(got<0):.3f}")
