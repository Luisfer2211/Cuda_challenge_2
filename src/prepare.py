"""Etapa 1: lectura, limpieza y preparacion del subconjunto de Gaia DR3.

    python src/prepare.py

Lee data/raw/*.fits, limpia, asigna teselas, estandariza las 5 variables por
tesela y elige eps por tesela. Escribe:
  data/clean/gaia_disk.parquet   catalogo limpio (todas las columnas + tesela)
  data/clean/features.npz        X (n,5) float32, tile (n,), eps (n_tiles,)
  results/prepare.json           conteos y tiempos de cada paso
"""
import glob
import json
import time

import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.neighbors import KDTree

import config as C


def read_raw() -> pd.DataFrame:
    files = sorted(glob.glob(str(C.DATA_RAW / "L*.fits")))
    parts = []
    for f in files:
        t = Table.read(f, format="fits")
        if len(t):
            parts.append(t.to_pandas())
    df = pd.concat(parts, ignore_index=True)
    return df, len(files)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quita duplicados y nulos en las 5 variables; normaliza tipos."""
    df = df.drop_duplicates("source_id")
    df = df.dropna(subset=C.FEATURES)
    df = df[(df["parallax"] > 0)]
    df["l"] = df["l"] % 360.0
    df["tile"] = (df["l"] // C.TILE_DEG).astype(np.int32)
    return df.reset_index(drop=True)


def standardize_cpu(df: pd.DataFrame) -> np.ndarray:
    """z-score de cada variable dentro de su tesela (NumPy/pandas)."""
    g = df.groupby("tile")[C.FEATURES]
    z = (df[C.FEATURES] - g.transform("mean")) / g.transform("std")
    return z.to_numpy(np.float32)


def standardize_gpu(X_raw: np.ndarray, tile: np.ndarray):
    """Mismo z-score con CuPy: suma por tesela con scatter-add en la GPU.
    Devuelve (X_std, tiempos) con la transferencia medida aparte."""
    import cupy as cp
    t = {"h2d": 0.0, "compute": 0.0, "d2h": 0.0}
    out = np.empty(X_raw.shape, np.float32)
    nt = int(tile.max()) + 1
    t0 = time.perf_counter()
    dt = cp.asarray(tile)
    cp.cuda.Device().synchronize()
    t["h2d"] += time.perf_counter() - t0
    cnt = cp.bincount(dt, minlength=nt).astype(cp.float64)
    # una columna a la vez para caber en los 4 GB de la RTX 3050
    for j in range(X_raw.shape[1]):
        col = np.ascontiguousarray(X_raw[:, j], dtype=np.float64)
        t0 = time.perf_counter()
        dx = cp.asarray(col)
        cp.cuda.Device().synchronize()
        t["h2d"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        s = cp.bincount(dt, weights=dx, minlength=nt)
        s2 = cp.bincount(dt, weights=dx * dx, minlength=nt)
        mean = s / cnt
        std = cp.sqrt((s2 - cnt * mean ** 2) / (cnt - 1))
        z = ((dx - mean[dt]) / std[dt]).astype(cp.float32)
        cp.cuda.Device().synchronize()
        t["compute"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        out[:, j] = cp.asnumpy(z)
        t["d2h"] += time.perf_counter() - t0
        del dx, z
    del dt
    cp.get_default_memory_pool().free_all_blocks()
    return out, t


def choose_eps(X: np.ndarray, tile: np.ndarray, k: int, pct: float, sample: int = 20_000,
               seed: int = 0) -> np.ndarray:
    """eps por tesela = percentil `pct` de la distancia al k-esimo vecino.

    Se calcula con una muestra de estrellas de cada tesela. Un percentil bajo
    significa que solo las zonas mucho mas densas que el campo forman nucleos.
    Asi eps se adapta a la densidad de cada tesela (el centro galactico es mucho
    mas denso que el anticentro).
    """
    rng = np.random.default_rng(seed)
    nt = int(tile.max()) + 1
    eps = np.zeros(nt, np.float32)
    for t in range(nt):
        Xt = X[tile == t]
        if len(Xt) <= k:
            eps[t] = 1e-3
            continue
        idx = rng.choice(len(Xt), size=min(sample, len(Xt)), replace=False)
        # la muestra se compara contra una submuestra del mismo tamano relativo
        # y se corrige la densidad: d_k(full) ~ d_k(sub) * (n_sub/n_full)^(1/5)
        sub = Xt[rng.choice(len(Xt), size=min(len(Xt), 200_000), replace=False)]
        tree = KDTree(sub)
        d, _ = tree.query(Xt[idx], k=k + 1)
        scale = (len(sub) / len(Xt)) ** (1 / 5)
        eps[t] = np.percentile(d[:, k], pct) * scale
    return eps


def main():
    C.DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    info = {}

    t0 = time.perf_counter()
    df, nfiles = read_raw()
    info["t_read_s"] = time.perf_counter() - t0
    info["raw_files"] = nfiles
    info["raw_rows"] = int(len(df))
    info["raw_bytes"] = int(sum(p.stat().st_size for p in C.DATA_RAW.glob("L*.fits")))

    t0 = time.perf_counter()
    df = clean(df)
    info["t_clean_s"] = time.perf_counter() - t0
    info["clean_rows"] = int(len(df))

    t0 = time.perf_counter()
    X = standardize_cpu(df)
    info["t_standardize_cpu_s"] = time.perf_counter() - t0
    tile = df["tile"].to_numpy(np.int32)
    try:
        Xg, tg = standardize_gpu(df[C.FEATURES].to_numpy(), tile)
        info["t_standardize_gpu_s"] = tg
        info["standardize_max_abs_diff"] = float(np.nanmax(np.abs(Xg - X)))
    except Exception as e:  # sin GPU: solo CPU
        info["t_standardize_gpu_s"] = f"no disponible: {e}"

    t0 = time.perf_counter()
    eps = choose_eps(X, tile, C.MIN_SAMPLES, C.EPS_PERCENTILE)
    info["t_choose_eps_s"] = time.perf_counter() - t0
    info["eps_per_tile"] = eps.round(4).tolist()
    info["stars_per_tile"] = np.bincount(tile).tolist()

    t0 = time.perf_counter()
    df.to_parquet(C.DATA_CLEAN / "gaia_disk.parquet", index=False)
    np.savez(C.DATA_CLEAN / "features.npz", X=X, tile=tile, eps=eps)
    info["t_write_s"] = time.perf_counter() - t0
    info["clean_bytes"] = int((C.DATA_CLEAN / "gaia_disk.parquet").stat().st_size)

    with open(C.RESULTS / "prepare.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k not in ("eps_per_tile", "stars_per_tile")},
                     indent=2))


if __name__ == "__main__":
    main()
