#!/usr/bin/env python3
"""
Merges two independently-built (matrix, species_stats, taxons) triples into
one — e.g. combining a matrix generated on one server (via
merge_fantasia_species.py in "fresh tables" mode) with the main matrix, or
combining two fresh batches before folding them into the main dataset.

Species present in BOTH inputs (matched by exact species name) are treated
as duplicates: one side wins per --on-duplicate, and every duplicate is
logged plus written to a report TSV so it can be reviewed manually — exact
name matching will not catch near-duplicates (e.g. a species recorded under
a strain-qualified name on one side and a plain binomial on the other).
"""

VERSION = "v0.1.0"

import argparse
import getpass
import gc
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

STATS_COLUMNS = [
    "Total_prots", "Total_mRNA", "Total_fan", "Total_hom",
    "Unique_fan", "Unique_hom", "Perc_fan", "Perc_hom",
    "GO/Prot_fan", "GO/Prot_hom", "IC_fan", "IC_hom",
]

# ------------------------------------------------------------------ logging
_LOG_FH = None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    if _LOG_FH is not None:
        print(line, file=_LOG_FH, flush=True)


def _banner(title: str) -> None:
    bar = "─" * (len(title) + 4)
    _log(f"┌{bar}┐")
    _log(f"│  {title}  │")
    _log(f"└{bar}┘")


def _load_matrix(path: Path) -> pd.DataFrame:
    header_cols = pd.read_csv(path, sep="\t", nrows=0).columns
    dtype_map = {c: "float32" for c in header_cols[1:]}
    df = pd.read_csv(path, sep="\t", index_col=0, dtype=dtype_map).fillna(0)
    return df.astype("int32")


def _validate_inputs(pairs):
    ok = True
    for flag, path in pairs:
        if not path.exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-a", type=Path, required=True)
    ap.add_argument("--stats-a", type=Path, required=True)
    ap.add_argument("--taxons-a", type=Path, required=True)
    ap.add_argument("--matrix-b", type=Path, required=True)
    ap.add_argument("--stats-b", type=Path, required=True)
    ap.add_argument("--taxons-b", type=Path, required=True)

    ap.add_argument("--matrix-out", type=Path, default=None, help="Default: overwrite --matrix-a (with a timestamped backup)")
    ap.add_argument("--stats-out", type=Path, default=None, help="Default: overwrite --stats-a (with a timestamped backup)")
    ap.add_argument("--taxons-out", type=Path, default=None, help="Default: overwrite --taxons-a (with a timestamped backup)")

    ap.add_argument("--on-duplicate", choices=["keep_a", "keep_b"], default="keep_a",
                     help="Species present in both inputs (exact name match): which side's row wins (default: keep_a)")

    ap.add_argument("--output", default="merge_two_matrices_run", help="Run directory for logs/results")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--disable_co2_tracking", action="store_true", help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    for attr in ("matrix_a", "stats_a", "taxons_a", "matrix_b", "stats_b", "taxons_b"):
        setattr(args, attr, getattr(args, attr).resolve())

    _validate_inputs([
        ("--matrix-a", args.matrix_a), ("--stats-a", args.stats_a), ("--taxons-a", args.taxons_a),
        ("--matrix-b", args.matrix_b), ("--stats-b", args.stats_b), ("--taxons-b", args.taxons_b),
    ])

    run_dir = Path(args.output)
    results = run_dir / "results"
    logs_dir = run_dir / "logs"
    for d in (results, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = logs_dir / "Run_MergeTwoMatrices.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  MergeTwoMatrices {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    matrix_out = (args.matrix_out or args.matrix_a).resolve()
    stats_out = (args.stats_out or args.stats_a).resolve()
    taxons_out = (args.taxons_out or args.taxons_a).resolve()

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  A: {args.matrix_a}, {args.stats_a}, {args.taxons_a}")
        _log(f"  B: {args.matrix_b}, {args.stats_b}, {args.taxons_b}")
        _log(f"  on-duplicate: {args.on_duplicate}")
        _log(f"  Matrix out  : {matrix_out}")
        _log(f"  Stats out   : {stats_out}")
        _log(f"  Taxons out  : {taxons_out}")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="MergeTwoMatrices", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped (conda install -c conda-forge codecarbon)")

    t_start = time.monotonic()

    _banner("Cargando matrices A y B")
    matrix_a = _load_matrix(args.matrix_a)
    matrix_b = _load_matrix(args.matrix_b)
    _log(f"  A: {matrix_a.shape[0]} especies x {matrix_a.shape[1]} GO terms ({args.matrix_a.name})")
    _log(f"  B: {matrix_b.shape[0]} especies x {matrix_b.shape[1]} GO terms ({args.matrix_b.name})")

    stats_a = pd.read_csv(args.stats_a, sep="\t")
    stats_b = pd.read_csv(args.stats_b, sep="\t")
    taxons_a = pd.read_csv(args.taxons_a, sep="\t")
    taxons_b = pd.read_csv(args.taxons_b, sep="\t")

    _banner("Detectando especies duplicadas (coincidencia exacta de nombre)")
    species_a = set(matrix_a.index)
    species_b = set(matrix_b.index)
    dup_species = species_a & species_b
    _log(f"  {len(dup_species)} especies presentes en A y B")

    if dup_species:
        group_a_map = dict(zip(taxons_a["Species"], taxons_a["Group"]))
        group_b_map = dict(zip(taxons_b["Species"], taxons_b["Group"]))
        dup_rows = []
        n_conflicts = 0
        for sp in sorted(dup_species):
            ga, gb = group_a_map.get(sp), group_b_map.get(sp)
            conflict = ga is not None and gb is not None and ga != gb
            if conflict:
                n_conflicts += 1
            dup_rows.append({"Species": sp, "Group_A": ga, "Group_B": gb, "Group_conflict": conflict,
                              "kept": "A" if args.on_duplicate == "keep_a" else "B"})
        dup_report = pd.DataFrame(dup_rows)
        dup_report.to_csv(results / f"duplicate_species_{prefix}.tsv", sep="\t", index=False)
        _log(f"  {n_conflicts} de esas especies tienen Group distinto en A vs B — revisar "
             f"results/duplicate_species_{prefix}.tsv")
        _log(f"  Política --on-duplicate={args.on_duplicate}: se descartan las filas del otro lado")

    if args.on_duplicate == "keep_a":
        matrix_b = matrix_b.drop(index=[s for s in dup_species if s in matrix_b.index])
        stats_b = stats_b[~stats_b["Species"].isin(dup_species)]
        taxons_b = taxons_b[~taxons_b["Species"].isin(dup_species)]
    else:
        matrix_a = matrix_a.drop(index=[s for s in dup_species if s in matrix_a.index])
        stats_a = stats_a[~stats_a["Species"].isin(dup_species)]
        taxons_a = taxons_a[~taxons_a["Species"].isin(dup_species)]

    _banner("Fusionando")
    all_go = matrix_a.columns.union(matrix_b.columns)
    n_new_go = len(all_go) - len(matrix_a.columns)
    _log(f"  {len(all_go)} GO terms totales tras la unión ({n_new_go} aportados solo por B)")
    matrix_a = matrix_a.reindex(columns=all_go, fill_value=0)
    matrix_b = matrix_b.reindex(columns=all_go, fill_value=0)
    merged_matrix = pd.concat([matrix_a, matrix_b], axis=0).astype("int32")
    del matrix_a, matrix_b
    gc.collect()

    merged_stats = pd.concat([stats_a, stats_b[["Species"] + STATS_COLUMNS]], ignore_index=True)
    merged_taxons = pd.concat([taxons_a, taxons_b[["Group", "Species"]]], ignore_index=True)

    n_total_species = merged_matrix.shape[0]
    _log(f"  Resultado: {n_total_species} especies x {merged_matrix.shape[1]} GO terms")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for live_path in (matrix_out, stats_out, taxons_out):
        if live_path.exists():
            backup = live_path.with_name(f"{live_path.name}.bak_{ts}")
            _log(f"  Backup {live_path.name} → {backup.name}")
            live_path.rename(backup)

    _log(f"  Escribiendo matriz fusionada → {matrix_out}")
    merged_matrix.to_csv(matrix_out, sep="\t")
    _log(f"  Escribiendo stats fusionadas ({len(merged_stats)} especies) → {stats_out}")
    merged_stats.to_csv(stats_out, sep="\t", index=False)
    _log(f"  Escribiendo taxonomía fusionada ({len(merged_taxons)} especies) → {taxons_out}")
    merged_taxons.to_csv(taxons_out, sep="\t", index=False)

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "n_species_a": len(species_a),
        "n_species_b": len(species_b),
        "n_duplicates": len(dup_species),
        "on_duplicate": args.on_duplicate,
        "n_new_go_terms_from_b": int(n_new_go),
        "n_total_species": int(n_total_species),
        "parameters": {
            "matrix_out": str(matrix_out),
            "stats_out": str(stats_out),
            "taxons_out": str(taxons_out),
        },
        "resource_usage": {
            "wall_clock_s": round(elapsed_s, 1),
            "peak_mem_mb": round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    with open(results / f"{prefix}.run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    _banner("Listo")
    _log(f"  Especies totales    : {n_total_species}")
    _log(f"  Duplicadas resueltas: {len(dup_species)} ({args.on_duplicate})")
    _log(f"  GO terms nuevos (de B): {n_new_go}")
    _log(f"  Tiempo total        : {elapsed_s:.1f}s")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
