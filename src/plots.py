"""Figuras para el informe y la presentacion (figures/*.png).

    python src/plots.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_speedup(bm):
    scales = list(bm["scales"])
    n = [bm["scales"][s]["n"] / 1e6 for s in scales]
    cpu = [bm["scales"][s]["t_cpu_s"] for s in scales]
    gpu = [bm["scales"][s]["t_gpu_s"] for s in scales]
    x = np.arange(len(scales))
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, cpu, w, color=ORANGE, label="CPU (scikit-learn, 12 hilos)")
    ax.bar(x + w / 2 + 0.01, gpu, w, color=BLUE, label="GPU (kernels CUDA)")
    ax.set_yscale("log")
    for i in range(len(scales)):
        ax.text(x[i] - w / 2, cpu[i] * 1.12, f"{cpu[i]:.0f} s", ha="center", fontsize=9, color=INK2)
        ax.text(x[i] + w / 2, gpu[i] * 1.12, f"{gpu[i]:.1f} s", ha="center", fontsize=9, color=INK2)
        ax.text(x[i], cpu[i] * 3.2, f"{cpu[i]/gpu[i]:.0f}× más rápido", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x, [f"{s}\n{v:.1f} M estrellas" for s, v in zip(scales, n)])
    ax.set_ylabel("tiempo de DBSCAN (s, escala log)")
    ax.set_ylim(min(gpu) / 3, max(cpu) * 8)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    ax.set_title("DBSCAN 5D sobre Gaia DR3: CPU vs GPU", loc="left", fontweight="bold")
    fig.savefig(C.FIGURES / "speedup.png")
    plt.close(fig)


def fig_gpu_stages(bm):
    sc = bm["scales"]["10x"]
    s = dict(sc["gpu_stages_s"])
    s["host"] = sc["t_gpu_s"] - sum(s.values())   # trabajo en CPU: armar lotes, indices
    names = {"host": "CPU: armar lotes (4 GB)", "transfer_h2d": "copia CPU→GPU", "grid_sort": "rejilla + ordenar",
             "core_points": "contar vecinos (núcleos)", "union_find": "union-find",
             "border_labels": "bordes + etiquetas", "transfer_d2h": "copia GPU→CPU"}
    keys = list(names)
    vals = [s[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ORANGE if k.startswith("transfer") else ("#9a9993" if k == "host" else BLUE)
              for k in keys]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    tot = sum(vals)
    for yi, v in zip(y, vals):
        ax.text(v + tot * 0.01, yi, f"{v:.2f} s ({100*v/tot:.0f}%)", va="center", fontsize=9,
                color=INK2)
    ax.set_yticks(y, [names[k] for k in keys])
    ax.set_xlabel(f"segundos (10x, 38.6 M estrellas, total {tot:.1f} s)\n"
                  "azul = kernels CUDA · naranja = copias PCIe · gris = trabajo en CPU")
    ax.set_xlim(0, max(vals) * 1.4)
    ax.set_title("¿En qué se va el tiempo de la GPU?", loc="left", fontweight="bold")
    fig.savefig(C.FIGURES / "gpu_stages.png")
    plt.close(fig)


def fig_amdahl(prep, bm):
    """Pipeline completo: etapas que no se paralelizan vs DBSCAN."""
    sc = bm["scales"]["10x"]
    std_gpu = prep["t_standardize_gpu_s"]
    std_gpu = sum(std_gpu.values()) if isinstance(std_gpu, dict) else prep["t_standardize_cpu_s"]
    common = [("leer FITS", prep["t_read_s"]), ("limpiar", prep["t_clean_s"]),
              ("elegir eps", prep["t_choose_eps_s"]), ("cruce catálogo", bm["t_validate_s"])]
    rows = {
        "CPU": common + [("estandarizar", prep["t_standardize_cpu_s"]), ("DBSCAN", sc["t_cpu_s"])],
        "GPU": common + [("estandarizar", std_gpu), ("DBSCAN", sc["t_gpu_s"])],
    }
    fig, ax = plt.subplots(figsize=(8, 2.8))
    palette = {"leer FITS": "#b9b8b2", "limpiar": "#9a9993", "elegir eps": "#7d7c76",
               "cruce catálogo": "#5f5e59", "estandarizar": VIOLET, "DBSCAN": None}
    for yi, (label, parts) in enumerate(rows.items()):
        left = 0
        for name, v in parts:
            c = palette[name] or (ORANGE if label == "CPU" else BLUE)
            ax.barh(yi, v, left=left, color=c, height=0.55, edgecolor="white", linewidth=1.5,
                    label=name if yi == 0 else None)
            left += v
        ax.text(left + 5, yi, f"{left:.0f} s", va="center", fontweight="bold")
    ax.set_yticks([0, 1], list(rows))
    ax.invert_yaxis()
    ax.set_xlabel("segundos, pipeline completo con 10x (todo el disco)")
    ax.legend(ncol=6, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Ley de Amdahl: lo que la GPU no acelera", loc="left", fontweight="bold")
    fig.savefig(C.FIGURES / "amdahl.png")
    plt.close(fig)


def fig_sky(clusters, df):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    sample = df.sample(min(len(df), 400_000), random_state=0)
    ax.hexbin(np.where(sample.l > 180, sample.l - 360, sample.l), sample.b, gridsize=(220, 25),
              bins="log", cmap="Greys", mincnt=1, linewidths=0, alpha=0.45)
    lw = np.where(clusters.l > 180, clusters.l - 360, clusters.l)
    known = clusters.match != ""
    ax.scatter(lw[~known], clusters.b[~known], s=7, color=ORANGE, label="candidato nuevo",
               edgecolor="white", linewidth=0.3)
    ax.scatter(lw[known], clusters.b[known], s=9, color=BLUE, label="cúmulo conocido (CG2020)",
               edgecolor="white", linewidth=0.3)
    ax.set_xlim(180, -180)
    ax.set_xlabel("longitud galáctica l (°)")
    ax.set_ylabel("b (°)")
    ax.grid(False)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(1, 1.22))
    ax.set_title(f"{len(clusters)} grupos en el disco: {known.sum()} conocidos, "
                 f"{(~known).sum()} candidatos", loc="left", fontweight="bold")
    fig.savefig(C.FIGURES / "sky_map.png")
    plt.close(fig)


def fig_example(clusters, df, labels, name="NGC_7789"):
    """Diagrama color-magnitud y movimiento propio de un cumulo conocido."""
    known = clusters[clusters.match == name]
    if known.empty:
        known = clusters[clusters.match != ""].sort_values("n", ascending=False)
    if known.empty:
        return
    row = known.iloc[0]
    members = df[labels == row.label]
    field = df[(np.abs(df.l - row.l) < 0.5) & (np.abs(df.b - row.b) < 0.5)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].scatter(field.pmra, field.pmdec, s=1, color="#c3c2b7", label="todas las estrellas (±0.5°)")
    axs[0].scatter(members.pmra, members.pmdec, s=6, color=BLUE, label="miembros DBSCAN")
    axs[0].set_xlim(row.pmra - 8, row.pmra + 8)
    axs[0].set_ylim(row.pmdec - 8, row.pmdec + 8)
    axs[0].set_xlabel("pmra (mas/año)")
    axs[0].set_ylabel("pmdec (mas/año)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].scatter(field.bp_rp, field.phot_g_mean_mag, s=1, color="#c3c2b7")
    axs[1].scatter(members.bp_rp, members.phot_g_mean_mag, s=6, color=BLUE)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("color BP−RP")
    axs[1].set_ylabel("magnitud G")
    fig.suptitle(f"{row.match.replace('_', ' ')}: {int(row.n)} estrellas que se mueven juntas "
                 f"y forman una sola secuencia de edad", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "example_cluster.png")
    plt.close(fig)


def main():
    C.FIGURES.mkdir(exist_ok=True)
    bm = json.load(open(C.RESULTS / "benchmark.json"))
    prep = json.load(open(C.RESULTS / "prepare.json"))
    fig_speedup(bm)
    fig_gpu_stages(bm)
    fig_amdahl(prep, bm)
    clusters = pd.read_csv(C.RESULTS / "clusters.csv", keep_default_na=False)
    clusters["match"] = clusters["match"].astype(str)
    df = pd.read_parquet(C.DATA_CLEAN / "gaia_disk.parquet",
                         columns=["l", "b", "pmra", "pmdec", "bp_rp", "phot_g_mean_mag"])
    fig_sky(clusters, df)
    fig_example(clusters, df, np.load(C.DATA_CLEAN / "labels_gpu.npy"))
    print("Figuras en", C.FIGURES)


if __name__ == "__main__":
    main()
