"""Descarga del subconjunto de Gaia DR3 por bloques de source_id (HEALPix).

Uso:
    python src/gaia_download.py count      # COUNT(*) por pixel HEALPix nivel 2
    python src/gaia_download.py download   # descarga por pixeles nivel 4 (FITS)

source_id es un indice del archivo, asi que filtrar por un rango de source_id
(= un pixel HEALPix) es rapido. Cada bloque se guarda en data/raw/L4_XXXXXXX.fits;
si ya existe se salta, asi que la descarga se puede reanudar.
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config as C

SESSION = requests.Session()


def where_clause(chunk, lo=None, hi=None) -> str:
    if chunk is not None:
        lo = chunk * C.CHUNK_SPAN
        hi = (chunk + 1) * C.CHUNK_SPAN - 1
    return (
        f"source_id BETWEEN {lo} AND {hi} "
        f"AND b BETWEEN {-C.B_MAX} AND {C.B_MAX} "
        f"AND parallax_over_error > {C.PARALLAX_OVER_ERROR} "
        f"AND ruwe < {C.RUWE_MAX} "
        f"AND phot_g_mean_mag < {C.G_MAX} "
        "AND pmra IS NOT NULL AND pmdec IS NOT NULL"
    )


def sync_query(adql: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            r = SESSION.post(f"{C.TAP_URL}/sync", timeout=120, data={
                "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
            if r.ok and "QUERY_STATUS" not in r.text[:500]:
                return r.text
            err = r.text[:300]
        except requests.RequestException as e:
            err = str(e)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Consulta fallida: {err}")


def count_chunk(chunk: int) -> int:
    txt = sync_query(f"SELECT COUNT(*) FROM gaiadr3.gaia_source WHERE {where_clause(chunk)}")
    return int(txt.strip().splitlines()[1])


def chunk_range(level: int, pix: int):
    span = 2 ** 35 * 4 ** (12 - level)
    return pix * span, (pix + 1) * span - 1


def fetch_pixel(level: int, pix: int, max_level: int = 7) -> str:
    """Descarga un pixel HEALPix (consulta sincrona, FITS binario).

    Los usuarios anonimos tienen un limite de 60 s por consulta. Si el pixel no
    alcanza a terminar, se divide en sus 4 hijos (nivel + 1) y se reintenta.
    """
    out = C.DATA_RAW / f"L{level}_{pix:07d}.fits"
    if out.exists():
        return f"{out.name} ya existe"
    lo, hi = chunk_range(level, pix)
    adql = (f"SELECT {', '.join(C.COLUMNS)} FROM gaiadr3.gaia_source "
            f"WHERE {where_clause(None, lo, hi)}")
    for attempt in range(3):
        t0 = time.time()
        try:
            r = SESSION.post(f"{C.TAP_URL}/sync", timeout=180, data={
                "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "fits", "MAXREC": "3000000", "QUERY": adql})
        except requests.RequestException as e:
            err = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        if r.ok and r.content[:6] == b"SIMPLE":
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(out)
            return f"{out.name}: {len(r.content)/1e6:.1f} MB en {time.time()-t0:.1f} s"
        err = r.text[:400]
        if "timeout" in err.lower() or "aborted" in err.lower():
            break  # demasiado grande: dividir
        time.sleep(5 * (attempt + 1))
    if level >= max_level:
        raise RuntimeError(f"L{level} pix {pix}: {err}")
    msgs = [fetch_pixel(level + 1, 4 * pix + k, max_level) for k in range(4)]
    return f"L{level}_{pix} dividido -> " + "; ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["count", "download"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mode == "count":
        counts = {}
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(count_chunk, c): c for c in range(C.N_CHUNKS)}
            for f in as_completed(futs):
                counts[futs[f]] = f.result()
        path = C.RESULTS / "chunk_counts.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["chunk", "n"])
            for c in sorted(counts):
                w.writerow([c, counts[c]])
        total = sum(counts.values())
        nonempty = sum(1 for v in counts.values() if v > 0)
        print(f"Total: {total:,} estrellas en {nonempty} bloques con datos "
              f"({time.time()-t0:.0f} s). Guardado en {path}")
    else:
        path = C.RESULTS / "chunk_counts.csv"
        parents = list(range(C.N_CHUNKS))
        if path.exists():  # solo bloques nivel 2 con estrellas
            rows = list(csv.DictReader(open(path)))
            parents = [int(r["chunk"]) for r in rows if int(r["n"]) > 0]
        # cada pixel nivel 2 se baja como 16 pixeles nivel 4 (bloques mas pequenos)
        sub = 4 ** (C.DOWNLOAD_LEVEL - C.HPX_LEVEL)
        chunks = [p * sub + k for p in parents for k in range(sub)]
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_pixel, C.DOWNLOAD_LEVEL, c): c for c in chunks}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    print(f"[{i}/{len(chunks)}] {f.result()}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{i}/{len(chunks)}] ERROR {e}", file=sys.stderr, flush=True)
        print(f"Descarga terminada en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
