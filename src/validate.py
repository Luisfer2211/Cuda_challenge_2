"""Resumen de cumulos encontrados y cruce con el catalogo de Cantat-Gaudin et al. (2020).

Un cumulo DBSCAN se considera "conocido" si su centro esta a menos de
max(2*r50, 0.5 grados) de un cumulo del catalogo, con |delta paralaje| < 0.3 mas
y |delta movimiento propio| < 0.7 mas/yr.
"""
import numpy as np
import pandas as pd

import config as C

CATALOG = C.ROOT / "data" / "catalogs" / "cantat_gaudin_2020.tsv"


def load_catalog() -> pd.DataFrame:
    lines = [ln for ln in open(CATALOG, encoding="utf-8") if not ln.startswith("#") and ln.strip()]
    rows = [ln.rstrip("\n").split("\t") for ln in lines[3:]]   # header, unidades, guiones
    cat = pd.DataFrame(rows, columns=[c.strip() for c in lines[0].rstrip("\n").split("\t")])
    cat["Cluster"] = cat["Cluster"].str.strip()
    for c in cat.columns[1:]:
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    return cat.rename(columns={"pmRA*": "pmRA"})


def summarize(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.assign(label=labels)
    d = d[d.label >= 0]
    # l puede cruzar 0/360 dentro de un cumulo: se promedia como angulo
    d = d.assign(lx=np.cos(np.radians(d.l)), ly=np.sin(np.radians(d.l)))
    g = d.groupby("label")
    s = g.agg(n=("l", "size"), lx=("lx", "mean"), ly=("ly", "mean"), b=("b", "mean"),
              plx=("parallax", "mean"), pmra=("pmra", "mean"), pmdec=("pmdec", "mean"),
              s_b=("b", "std"), s_pmra=("pmra", "std"), s_pmdec=("pmdec", "std"),
              s_plx=("parallax", "std"), tile=("tile", "first"))
    s["l"] = np.degrees(np.arctan2(s.ly, s.lx)) % 360
    return s.drop(columns=["lx", "ly"]).reset_index()


def crossmatch(summary: pd.DataFrame, cat: pd.DataFrame | None = None) -> pd.DataFrame:
    cat = load_catalog() if cat is None else cat
    cat = cat[cat.GLAT.abs() < C.B_MAX + 1]
    l1, b1 = np.radians(summary.l.to_numpy())[:, None], np.radians(summary.b.to_numpy())[:, None]
    l2, b2 = np.radians(cat.GLON.to_numpy())[None], np.radians(cat.GLAT.to_numpy())[None]
    cos_sep = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2)
    sep = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))
    dplx = np.abs(summary.plx.to_numpy()[:, None] - cat.plx.to_numpy()[None])
    dpm = np.hypot(summary.pmra.to_numpy()[:, None] - cat.pmRA.to_numpy()[None],
                   summary.pmdec.to_numpy()[:, None] - cat.pmDE.to_numpy()[None])
    rad = np.maximum(2 * cat.r50.to_numpy(), 0.5)[None]
    ok = (sep < rad) & (dplx < 0.3) & (dpm < 0.7)
    score = np.where(ok, sep / rad + dplx / 0.3 + dpm / 0.7, np.inf)
    best = score.argmin(axis=1)
    has = np.isfinite(score[np.arange(len(summary)), best])
    out = summary.copy()
    out["match"] = np.where(has, cat.Cluster.to_numpy()[best], "")
    out["match_sep_deg"] = np.where(has, sep[np.arange(len(summary)), best], np.nan)
    return out


def report(matched: pd.DataFrame, cat: pd.DataFrame | None = None, region=None) -> dict:
    """Cuantos cumulos del catalogo (en la region) se recuperan y cuantos candidatos nuevos."""
    cat = load_catalog() if cat is None else cat
    sel = cat.GLAT.abs() < C.B_MAX
    if region is not None:
        sel &= cat.GLON.between(*region)
    cat_in = cat[sel & (cat.plx > 0)]
    recovered = set(matched.match[matched.match != ""]) & set(cat_in.Cluster)
    return {
        "clusters_found": int(len(matched)),
        "matched_known": int((matched.match != "").sum()),
        "known_in_region": int(len(cat_in)),
        "known_recovered": int(len(recovered)),
        "candidates_new": int((matched.match == "").sum()),
    }
