"""Etapas 2 y 3: DBSCAN 5D en CPU y en GPU con los mismos datos y parametros.

    python src/benchmark.py            # todo (CPU ~ decenas de minutos)
    python src/benchmark.py --no-cpu   # solo GPU (reusa results/cpu_tile_times.json)

Mide, para 1x, 3x y 10x de datos (config.SCALES):
  T_CPU   scikit-learn DBSCAN por tesela (KD-tree, 12 hilos)
  T_GPU   kernels CUDA propios, desglosado en transferencia H2D, rejilla+orden,
          nucleos, union-find, bordes y transferencia D2H (eventos CUDA)
  ARI     Adjusted Rand Index entre etiquetas CPU y GPU
Guarda results/benchmark.json, results/clusters.csv y data/clean/labels_*.npy.
"""
import argparse
import json
import platform
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

import config as C
import dbscan_cpu
import validate as V

GPU_REPEATS = 5


def gpu_info():
    import cupy as cp
    p = cp.cuda.runtime.getDeviceProperties(0)
    return {"name": p["name"].decode(), "mem_gb": round(p["totalGlobalMem"] / 2**30, 1),
            "sms": p["multiProcessorCount"], "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cupy": cp.__version__}


def cpu_info():
    import os
    import sklearn
    return {"processor": platform.processor(), "threads": os.cpu_count(),
            "python": platform.python_version(), "sklearn": sklearn.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu", action="store_true")
    args = ap.parse_args()

    import dbscan_gpu as G

    f = np.load(C.DATA_CLEAN / "features.npz")
    X, tile, eps = f["X"], f["tile"], f["eps"]
    out = {"n_total": int(len(X)), "min_samples": C.MIN_SAMPLES,
           "eps_percentile": C.EPS_PERCENTILE, "gpu": gpu_info(), "cpu": cpu_info(), "scales": {}}
    print(f"{len(X):,} estrellas, {len(eps)} teselas. GPU: {out['gpu']['name']}")

    # ---- CPU: todas las teselas una vez; las escalas suman sus teselas ----
    cpu_times_path = C.RESULTS / "cpu_tile_times.json"
    if args.no_cpu:
        cpu_times = {int(k): v for k, v in json.load(open(cpu_times_path)).items()}
        lab_cpu = np.load(C.DATA_CLEAN / "labels_cpu.npy")
    else:
        print("CPU (scikit-learn):")
        t0 = time.perf_counter()
        lab_cpu, cpu_times = dbscan_cpu.dbscan_cpu(X, tile, eps, C.MIN_SAMPLES, verbose=True)
        print(f"CPU total {time.perf_counter()-t0:.1f} s")
        np.save(C.DATA_CLEAN / "labels_cpu.npy", lab_cpu)
        json.dump(cpu_times, open(cpu_times_path, "w"), indent=1)

    # ---- GPU ----
    G.warmup()
    for name, tiles in C.SCALES.items():
        m = np.isin(tile, tiles)
        Xs, ts = X[m], tile[m]
        runs = []
        for r in range(GPU_REPEATS):
            timer = G.Timer()
            t0 = time.perf_counter()
            lab_gpu, nb = G.dbscan_gpu_batched(Xs, ts, eps, C.MIN_SAMPLES, C.GPU_MAX_POINTS, timer)
            wall = time.perf_counter() - t0
            runs.append((wall, timer.times))
        walls = [w for w, _ in runs]
        med = int(np.argsort(walls)[len(walls) // 2])
        stages = runs[med][1]
        t_cpu = sum(cpu_times[t] for t in tiles)
        t_gpu = walls[med]
        transfer = stages["transfer_h2d"] + stages["transfer_d2h"]
        ari = adjusted_rand_score(lab_cpu[m], lab_gpu)
        res = {
            "tiles": tiles, "n": int(m.sum()), "gpu_batches": nb,
            "t_cpu_s": t_cpu, "t_gpu_s": t_gpu, "t_gpu_runs_s": walls,
            "t_gpu_kernels_s": sum(v for k, v in stages.items() if not k.startswith("transfer")),
            "t_transfer_s": transfer, "transfer_frac": transfer / t_gpu,
            "gpu_stages_s": stages,
            "speedup": t_cpu / t_gpu, "speedup_without_transfer": t_cpu / (t_gpu - transfer),
            "ari_cpu_vs_gpu": ari,
            "clusters_cpu": int(len(np.unique(lab_cpu[m][lab_cpu[m] >= 0]))),
            "clusters_gpu": int(len(np.unique(lab_gpu[lab_gpu >= 0]))),
            "noise_agreement": float(np.mean((lab_cpu[m] < 0) == (lab_gpu < 0))),
        }
        out["scales"][name] = res
        print(f"{name:>3}: n={res['n']:>11,}  T_CPU={t_cpu:8.2f}s  T_GPU={t_gpu:6.3f}s  "
              f"(transfer {transfer:.3f}s = {100*res['transfer_frac']:.0f}%)  "
              f"speedup={res['speedup']:6.1f}x  ARI={ari:.5f}  "
              f"grupos cpu/gpu={res['clusters_cpu']}/{res['clusters_gpu']}", flush=True)
        if name == "10x":
            np.save(C.DATA_CLEAN / "labels_gpu.npy", lab_gpu)

    # ---- Validacion cientifica con el resultado completo (GPU) ----
    import pandas as pd
    df = pd.read_parquet(C.DATA_CLEAN / "gaia_disk.parquet")
    t0 = time.perf_counter()
    lab = np.load(C.DATA_CLEAN / "labels_gpu.npy")
    s = V.summarize(df, lab)
    s = s[(s.n >= C.CANDIDATE_MIN) & (s.n <= C.CANDIDATE_MAX)]
    matched = V.crossmatch(s)
    out["t_validate_s"] = time.perf_counter() - t0
    out["validation"] = V.report(matched)
    matched.to_csv(C.RESULTS / "clusters.csv", index=False)
    print("Validacion:", out["validation"])

    json.dump(out, open(C.RESULTS / "benchmark.json", "w"), indent=2)


if __name__ == "__main__":
    main()
