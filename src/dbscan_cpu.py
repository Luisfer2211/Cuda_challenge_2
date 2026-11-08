"""DBSCAN en CPU (linea base): scikit-learn, una tesela a la vez.

scikit-learn usa un KD-tree para buscar vecinos (paralelo en todos los nucleos
con n_jobs=-1) y luego expande los cumulos en un solo hilo. Es la implementacion
de referencia que usaria un astronomo en su computadora.
"""
import time

import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_cpu(X, tile, eps, min_samples, tiles=None, verbose=False):
    """Devuelve (labels globales, tiempo por tesela en s). Ruido = -1."""
    labels = np.full(len(X), -1, dtype=np.int32)
    times = {}
    offset = 0
    for t in (np.unique(tile) if tiles is None else tiles):
        idx = np.flatnonzero(tile == t)
        if len(idx) == 0:
            continue
        Xt = X[idx].astype(np.float64)
        t0 = time.perf_counter()
        lab = DBSCAN(eps=float(eps[t]), min_samples=min_samples,
                     algorithm="kd_tree", n_jobs=-1).fit_predict(Xt)
        times[int(t)] = time.perf_counter() - t0
        labels[idx] = np.where(lab >= 0, lab + offset, -1)
        offset += lab.max() + 1 if lab.max() >= 0 else 0
        if verbose:
            print(f"  tesela {t:2d}: {len(idx):>9,} estrellas, {times[int(t)]:7.2f} s, "
                  f"{lab.max()+1} grupos", flush=True)
    return labels, times
