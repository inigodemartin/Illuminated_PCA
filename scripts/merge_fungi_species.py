#!/usr/bin/env python3
"""
Ingests newly-run Fungi species from the FANTASIA_project non_viridiplantae
tree and merges them into the existing species x GO-term counts matrix,
species stats and taxonomy tables -- skipping any species already present.

Expected layout per species (see fungi_structure.txt):

    {base_dir}/{Species_Name}/
      04_FunctionalAnnotation/
        FANTASIA_2025_{assembly}/{code}_GOs_merged.tsv         -- FANTASIA GO calls
        Homology_annot_{assembly}/{code}.proteins.funct_ahrd.tsv -- AHRD homology annotation

The species directory name IS the species name used everywhere else in this
project (e.g. "Aaosphaeria_arxii_CBS_175.79") -- no metadata file needed to
map codes to names, unlike merge_fantasia_species.py.

Both GOs_merged.tsv and funct_ahrd.tsv are protein_id -> comma-separated GO
list, one row per protein (same shape as data/ASG001_topgo.txt), but they
differ in coverage: GOs_merged.tsv only lists proteins FANTASIA managed to
annotate, while funct_ahrd.tsv has one row per protein in the whole
predicted proteome (GO column empty when AHRD found no hit) -- so, when
present, the AHRD file's row count is used as the TRUE Total_prots (proteome
size), not an approximation. Total_mRNA is set equal to Total_prots, matching
the existing rows in merged_species_stats.tsv (one longest-isoform
transcript per gene, already selected upstream -- see the genomic_longest_
isof file names in fungi_structure.txt). When a species has no
Homology_annot_* output, Total_prots falls back to the FANTASIA-annotated
protein count (the same approximation merge_fantasia_species.py uses for
Asgard/Metazoa/Protists), Total_mRNA/*_hom columns stay NaN, and this is
logged so it's not silently mistaken for a real proteome size.
"""

VERSION = "v0.1.0"

import argparse
import gc
import getpass
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from general_pca_common import DEFAULT_IC_PATH, load_go_ic  # noqa: E402

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


def _checkpoint(path: Path, label: str, force: bool) -> bool:
    if not force and path.exists() and path.stat().st_size > 0:
        _log(f"  [checkpoint] {label} — {path.name} already exists, skipping")
        return True
    return False


# ------------------------------------------------------------- file parsing
def parse_go_list_file(path: Path, ic_map: dict, go_col: int = 1, skip_prefixes=("#",), header_prefix=None):
    """
    Generic protein_id \\t ... \\t comma-separated-GO-list parser, shared by
    GOs_merged.tsv (go_col=1, no header) and funct_ahrd.tsv (go_col=5, AHRD
    header). Returns (go_counter, n_rows, n_annotated_proteins,
    total_go_instances, ic_mean).

    n_rows is every data row seen (proteins with and without GO calls);
    n_annotated_proteins only counts rows that have at least one GO term.
    """
    go_counter = Counter()
    n_rows = 0
    n_annotated = 0
    total_instances = 0
    per_protein_ics = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            if header_prefix is not None and line.startswith(header_prefix):
                continue
            parts = line.split("\t")
            if len(parts) <= go_col:
                n_rows += 1
                continue
            n_rows += 1
            gos = [g.strip() for g in parts[go_col].split(",") if g.strip()]
            if not gos:
                continue
            n_annotated += 1
            total_instances += len(gos)
            for g in gos:
                go_counter[g] += 1
            ics = [ic_map[g] for g in gos if g in ic_map]
            if ics:
                per_protein_ics.append(sum(ics) / len(ics))
    ic_mean = float(np.mean(per_protein_ics)) if per_protein_ics else np.nan
    return go_counter, n_rows, n_annotated, total_instances, ic_mean


def parse_fantasia_file(path: Path, ic_map: dict):
    return parse_go_list_file(path, ic_map, go_col=1, skip_prefixes=(), header_prefix=None)


def parse_ahrd_file(path: Path, ic_map: dict):
    return parse_go_list_file(path, ic_map, go_col=5, skip_prefixes=("#",), header_prefix="Protein-Accession\t")


# --------------------------------------------------------- species discovery
def discover_fungi(base_dir: Path):
    entries = []
    for sp_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        species = sp_dir.name
        fa_dir = sp_dir / "04_FunctionalAnnotation"
        fantasia_matches = sorted(fa_dir.glob("FANTASIA_2025*/*_GOs_merged.tsv"))
        if not fantasia_matches:
            _log(f"  [WARN] fungi: {species} sin *_GOs_merged.tsv en {fa_dir}, se omite")
            continue
        if len(fantasia_matches) > 1:
            _log(f"  [WARN] fungi: {species} tiene {len(fantasia_matches)} ficheros GOs_merged.tsv, "
                 f"usando el primero ({fantasia_matches[0].name})")
        ahrd_matches = sorted(fa_dir.glob("Homology_annot_*/*.proteins.funct_ahrd.tsv"))
        ahrd_path = None
        if ahrd_matches:
            if len(ahrd_matches) > 1:
                _log(f"  [WARN] fungi: {species} tiene {len(ahrd_matches)} ficheros funct_ahrd.tsv, "
                     f"usando el primero ({ahrd_matches[0].name})")
            ahrd_path = ahrd_matches[0]
        entries.append({"species": species, "fantasia_path": fantasia_matches[0], "ahrd_path": ahrd_path})
    return entries


# ------------------------------------------------------------- group runner
def process_fungi(entries: list, ic_map: dict, workdir: Path, force: bool):
    long_path = workdir / "fungi_long.tsv.gz"
    stats_path = workdir / "fungi_stats.tsv"
    taxons_path = workdir / "fungi_taxons.tsv"

    if (_checkpoint(long_path, "fungi counts", force)
            and _checkpoint(stats_path, "fungi stats", force)
            and _checkpoint(taxons_path, "fungi taxons", force)):
        return

    _log(f"  Parsing {len(entries)} especies de fungi ...")
    long_rows = []
    stats_rows = []
    taxon_rows = []
    n_with_ahrd = 0
    t0 = time.monotonic()
    for i, e in enumerate(entries, 1):
        go_counter, _, n_annotated_fan, total_fan, ic_fan = parse_fantasia_file(e["fantasia_path"], ic_map)
        if n_annotated_fan == 0:
            _log(f"  [WARN] {e['species']}: 0 proteínas anotadas por FANTASIA, se omite")
            continue

        if e["ahrd_path"] is not None:
            hom_counter, n_prots, n_annotated_hom, total_hom, ic_hom = parse_ahrd_file(e["ahrd_path"], ic_map)
            total_prots = n_prots
            total_mrna = n_prots
            unique_hom = len(hom_counter)
            perc_hom = n_annotated_hom / n_prots if n_prots > 0 else np.nan
            go_per_prot_hom = (total_hom / n_annotated_hom) if n_annotated_hom > 0 else np.nan
            n_with_ahrd += 1
        else:
            _log(f"  [WARN] {e['species']}: sin Homology_annot_*, Total_prots se aproxima "
                 f"al nº de proteínas anotadas por FANTASIA")
            total_prots = n_annotated_fan
            total_mrna = n_annotated_fan
            unique_hom = np.nan
            perc_hom = np.nan
            go_per_prot_hom = np.nan
            total_hom = np.nan
            ic_hom = np.nan

        for go_id, count in go_counter.items():
            long_rows.append((e["species"], go_id, count))

        stats_rows.append({
            "Species": e["species"],
            "Total_prots": total_prots,
            "Total_mRNA": total_mrna,
            "Total_fan": total_fan,
            "Total_hom": total_hom,
            "Unique_fan": len(go_counter),
            "Unique_hom": unique_hom,
            "Perc_fan": n_annotated_fan / total_prots if total_prots > 0 else np.nan,
            "Perc_hom": perc_hom,
            "GO/Prot_fan": total_fan / n_annotated_fan if n_annotated_fan > 0 else np.nan,
            "GO/Prot_hom": go_per_prot_hom,
            "IC_fan": ic_fan,
            "IC_hom": ic_hom,
        })
        taxon_rows.append({"Group": "Fungi", "Species": e["species"]})
        if i % 200 == 0:
            _log(f"    ... {i}/{len(entries)} procesados ({time.monotonic()-t0:.0f}s)")

    long_df = pd.DataFrame(long_rows, columns=["Species", "GO", "Count"])
    long_df.to_csv(long_path, sep="\t", index=False, compression="gzip")

    stats_df = pd.DataFrame(stats_rows)[["Species"] + STATS_COLUMNS]
    stats_df.to_csv(stats_path, sep="\t", index=False)

    taxons_df = pd.DataFrame(taxon_rows)[["Group", "Species"]]
    taxons_df.to_csv(taxons_path, sep="\t", index=False)

    _log(f"  fungi: {len(stats_rows)} especies con anotación ({n_with_ahrd} con Total_prots real vía AHRD, "
         f"{len(stats_rows) - n_with_ahrd} aproximado), {long_df['GO'].nunique()} GO terms distintos")


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fungi-dir", type=Path, required=True,
                     help="non_viridiplantae_species base dir (one subdirectory per species, see fungi_structure.txt)")

    ap.add_argument("--current-matrix", type=Path, default=Path("merged_PCA_fantasia.tsv"))
    ap.add_argument("--current-stats", type=Path, default=Path("merged_species_stats.tsv"))
    ap.add_argument("--current-taxons", type=Path, default=Path("merged_taxons.tsv"))
    ap.add_argument("--ic-file", type=Path, default=DEFAULT_IC_PATH, help="GO id -> IC TSV (default: bundled data/All_GOs_ic.tsv)")

    ap.add_argument("--matrix-out", type=Path, default=None, help="Default: overwrite --current-matrix (with a timestamped backup)")
    ap.add_argument("--stats-out", type=Path, default=None, help="Default: overwrite --current-stats (with a timestamped backup)")
    ap.add_argument("--taxons-out", type=Path, default=None, help="Default: overwrite --current-taxons (with a timestamped backup)")

    ap.add_argument("--output", default="merge_fungi_run", help="Run directory for logs/workdir/results")

    ap.add_argument("--skip_merge", action="store_true", help="Parse/checkpoint the fungi tables only; do not write the merged matrix/stats/taxons")
    ap.add_argument("--force", action="store_true", help="Rerun all steps from scratch even if intermediate outputs exist in workdir/")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--disable_co2_tracking", action="store_true", help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    args.fungi_dir = args.fungi_dir.resolve()
    args.current_matrix = args.current_matrix.resolve()
    args.current_stats = args.current_stats.resolve()
    args.current_taxons = args.current_taxons.resolve()
    args.ic_file = args.ic_file.resolve()

    if not args.fungi_dir.exists():
        print(f"ERROR: --fungi-dir not found: {args.fungi_dir}", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.output)
    results = run_dir / "results"
    workdir = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = logs_dir / "Run_MergeFungiSpecies.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  MergeFungiSpecies {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    matrix_out = (args.matrix_out or args.current_matrix).resolve()
    stats_out = (args.stats_out or args.current_stats).resolve()
    taxons_out = (args.taxons_out or args.current_taxons).resolve()

    has_existing_data = args.current_matrix.exists() and args.current_stats.exists() and args.current_taxons.exists()
    if not has_existing_data and (args.current_matrix.exists() or args.current_stats.exists() or args.current_taxons.exists()):
        print("ERROR: de --current-matrix/--current-stats/--current-taxons, algunos existen y otros no "
              "— o pasa las tres rutas de un dataset consistente, o ninguna (para generar tablas nuevas desde cero):",
              file=sys.stderr)
        for flag, p in (("--current-matrix", args.current_matrix), ("--current-stats", args.current_stats), ("--current-taxons", args.current_taxons)):
            print(f"  {flag}: {p} {'(existe)' if p.exists() else '(NO existe)'}", file=sys.stderr)
        sys.exit(1)

    has_ic_file = args.ic_file.exists()

    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif workdir.exists() and any(workdir.iterdir()):
        _log("Existing workdir found — resuming from checkpoints (use --force to rerun all steps from scratch)")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Fungi dir   : {args.fungi_dir}")
        _log(f"  Modo        : {'fusión con datos existentes' if has_existing_data else 'tablas nuevas desde cero (no se encontraron datos previos)'}")
        _log(f"  IC file     : {args.ic_file if has_ic_file else 'no encontrado — IC_fan/IC_hom quedarán vacíos'}")
        _log(f"  Matrix out  : {matrix_out}")
        _log(f"  Stats out   : {stats_out}")
        _log(f"  Taxons out  : {taxons_out}")
        _log("  Steps that would run:")
        _log("    [1] Ingest Fungi   → workdir/fungi_{long,stats,taxons}.*")
        if not args.skip_merge:
            _log("    [2] Merge into matrix/stats/taxons → live files above")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="MergeFungiSpecies", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped (conda install -c conda-forge codecarbon)")

    t_start = time.monotonic()

    _banner("Cargando IC de referencia")
    if has_ic_file:
        ic_map = load_go_ic(args.ic_file)
        _log(f"  {len(ic_map)} GO ids con IC cargados desde {args.ic_file.name}")
    else:
        ic_map = {}
        _log(f"  [WARN] --ic-file no encontrado ({args.ic_file}) — IC_fan/IC_hom quedarán vacíos")

    _banner("Módulo 1 — Fungi")
    entries = discover_fungi(args.fungi_dir)
    _log(f"  {len(entries)} especies con *_GOs_merged.tsv encontrado")
    process_fungi(entries, ic_map, workdir, args.force)

    n_new_species = 0
    n_new_go = 0
    n_skipped_existing = 0
    n_dup_matrix = 0
    n_dup_stats = 0
    n_dup_taxons = 0
    if not args.skip_merge:
        _banner("Módulo 2 — Fusión con la matriz/stats/taxons actuales" if has_existing_data
                 else "Módulo 2 — Construcción de tablas nuevas (sin datos previos)")

        if has_existing_data:
            _log(f"  Cargando matriz actual ({args.current_matrix.name}) ...")
            header_cols = pd.read_csv(args.current_matrix, sep="\t", nrows=0).columns
            dtype_map = {c: "float32" for c in header_cols[1:]}
            current_matrix = pd.read_csv(args.current_matrix, sep="\t", index_col=0, dtype=dtype_map).fillna(0)
            current_matrix = current_matrix.astype("int32")
            current_taxons = pd.read_csv(args.current_taxons, sep="\t")
            current_stats = pd.read_csv(args.current_stats, sep="\t")
        else:
            current_matrix = pd.DataFrame(dtype="int32")
            current_matrix.index.name = "Species"
            current_taxons = pd.DataFrame(columns=["Group", "Species"])
            current_stats = pd.DataFrame(columns=["Species"] + STATS_COLUMNS)

        long_df = pd.read_csv(workdir / "fungi_long.tsv.gz", sep="\t")
        stats_df = pd.read_csv(workdir / "fungi_stats.tsv", sep="\t")
        taxon_df = pd.read_csv(workdir / "fungi_taxons.tsv", sep="\t")

        # Each table is checked independently against its own "already
        # present" set. A species can be in current_taxons (e.g. seeded from
        # a broader roster, or left over from a prior partial run) without
        # having a row in current_matrix yet -- treating "known" as the union
        # of matrix+taxons (as this used to do) would then skip it from the
        # matrix forever, even though it still needs to be merged there.
        matrix_new_mask = ~stats_df["Species"].isin(set(current_matrix.index))
        stats_new_mask = ~stats_df["Species"].isin(set(current_stats["Species"]))
        taxons_new_mask = ~taxon_df["Species"].isin(set(current_taxons["Species"]))

        matrix_new_species = set(stats_df.loc[matrix_new_mask, "Species"])
        n_skipped_existing = int((~matrix_new_mask).sum())
        if n_skipped_existing:
            _log(f"  {n_skipped_existing} especies ya presentes en la matriz — se omiten de la matriz")
        n_stats_skipped = int((~stats_new_mask).sum())
        if n_stats_skipped:
            _log(f"  {n_stats_skipped} especies ya presentes en stats — se omiten de stats")
        n_taxons_skipped = int((~taxons_new_mask).sum())
        if n_taxons_skipped:
            _log(f"  {n_taxons_skipped} especies ya presentes en taxonomía — se omiten de taxonomía")

        long_df = long_df[long_df["Species"].isin(matrix_new_species)]
        stats_new_df = stats_df[stats_new_mask]
        taxon_new_df = taxon_df[taxons_new_mask]

        _log(f"  {len(matrix_new_species)} especies nuevas a añadir a la matriz "
             f"({len(stats_new_df)} a stats, {len(taxon_new_df)} a taxonomía)")
        n_new_species = len(matrix_new_species)

        if len(matrix_new_species) == 0 and len(stats_new_df) == 0 and len(taxon_new_df) == 0:
            _log("  Nada nuevo que fusionar.")
        else:
            if len(matrix_new_species) > 0:
                _log("  Pivotando conteos a formato ancho ...")
                new_wide = long_df.pivot_table(index="Species", columns="GO", values="Count", fill_value=0, aggfunc="sum")
                new_wide = new_wide.astype("int32")
                del long_df
                gc.collect()

                new_wide.to_csv(results / f"mod01_new_species_counts_{prefix}.tsv", sep="\t")

                all_go = current_matrix.columns.union(new_wide.columns)
                n_new_go = len(all_go) - len(current_matrix.columns)
                _log(f"  {n_new_go} GO terms nuevos no presentes en la matriz actual")

                current_matrix = current_matrix.reindex(columns=all_go, fill_value=0)
                new_wide = new_wide.reindex(columns=all_go, fill_value=0)
                merged_matrix = pd.concat([current_matrix, new_wide], axis=0).astype("int32")
                del current_matrix, new_wide
                gc.collect()
            else:
                _log("  Sin especies nuevas para la matriz — se mantiene igual.")
                merged_matrix = current_matrix

            merged_stats = pd.concat([current_stats, stats_new_df[["Species"] + STATS_COLUMNS]], ignore_index=True)
            merged_taxons = pd.concat([current_taxons, taxon_new_df[["Group", "Species"]]], ignore_index=True)

            n_dup_matrix = int(merged_matrix.index.duplicated().sum())
            if n_dup_matrix:
                _log(f"  {n_dup_matrix} filas duplicadas en la matriz — eliminando (se conserva la primera aparición)")
                merged_matrix = merged_matrix[~merged_matrix.index.duplicated(keep="first")]

            n_dup_stats = int(merged_stats["Species"].duplicated().sum())
            if n_dup_stats:
                _log(f"  {n_dup_stats} filas duplicadas en stats — eliminando (se conserva la primera aparición)")
                merged_stats = merged_stats.drop_duplicates(subset="Species", keep="first").reset_index(drop=True)

            n_dup_taxons = int(merged_taxons["Species"].duplicated().sum())
            if n_dup_taxons:
                _log(f"  {n_dup_taxons} filas duplicadas en taxonomía — eliminando (se conserva la primera aparición)")
                merged_taxons = merged_taxons.drop_duplicates(subset="Species", keep="first").reset_index(drop=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for live_path in (matrix_out, stats_out, taxons_out):
                if live_path.exists():
                    backup = live_path.with_name(f"{live_path.name}.bak_{ts}")
                    _log(f"  Backup {live_path.name} → {backup.name}")
                    live_path.rename(backup)

            _log(f"  Escribiendo matriz fusionada ({merged_matrix.shape[0]} x {merged_matrix.shape[1]}) → {matrix_out}")
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
        "n_new_species": int(n_new_species),
        "n_new_go_terms": int(n_new_go),
        "n_skipped_already_present": int(n_skipped_existing),
        "n_duplicates_removed": {
            "matrix": int(n_dup_matrix),
            "stats": int(n_dup_stats),
            "taxons": int(n_dup_taxons),
        },
        "parameters": {
            "matrix_out": str(matrix_out),
            "stats_out": str(stats_out),
            "taxons_out": str(taxons_out),
            "skip_merge": args.skip_merge,
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
    _log(f"  Especies nuevas añadidas : {n_new_species}")
    _log(f"  GO terms nuevos          : {n_new_go}")
    _log(f"  Especies omitidas (ya existían) : {n_skipped_existing}")
    if n_dup_matrix or n_dup_stats or n_dup_taxons:
        _log(f"  Duplicados eliminados    : matriz={n_dup_matrix}, stats={n_dup_stats}, taxons={n_dup_taxons}")
    _log(f"  Tiempo total             : {elapsed_s:.1f}s")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
