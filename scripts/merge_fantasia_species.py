#!/usr/bin/env python3
"""
Ingests new FANTASIA GO-annotation batches (Asgard archaea, Metazoa, extra
Protists) from their topgo tables + metadata, and merges them into the
existing species x GO-term counts matrix, species stats table and taxonomy
table.

Inputs per group are the raw topgo tables FANTASIA produces
(protein_id \\t GO:xxx, GO:yyy, ...) plus a metadata TSV mapping each file's
code to a species name / taxonomic group. See data/estructura_datos.txt for
the expected directory layout and data/ASG001_topgo.txt for the topgo format.

Group -> taxonomy tag rules (agreed with the user for this batch):
  - Asgard   : flat "Asgard" tag for every species (matches how "Fungi" is
               already a flat kingdom-level tag in merged_taxons.tsv).
  - Metazoa  : flat "Metazoa" tag for every species (same reasoning).
  - Protists : division column for supergroup=="Archaeplastida",
               subdivision column for supergroup=="Obazoa" -- this replicates
               the (inconsistent-looking but deliberate) precedent already
               present in merged_taxons.tsv for Rhodophyta/chlorophyta.
               Any other supergroup value falls back to division with a
               logged warning (none seen in this batch's metadata).

Total_prots caveat: topgo tables only list proteins that received at least
one GO annotation -- there is no whole-proteome count anywhere in the new
metadata. Total_prots for new species is therefore the number of annotated
proteins (same approximation the existing pipeline already uses for
Total_fan/Unique_fan), not a true proteome size. Perc_fan is trivially 1.0
as a result -- this is documented, not a bug.
"""

VERSION = "v0.1.0"

import argparse
import gc
import getpass
import gzip
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


# ------------------------------------------------------------- topgo parsing
def parse_topgo_file(path: Path, ic_map: dict):
    """
    One topgo file -> (Counter GO->n_proteins_with_it, n_annotated_proteins,
    total_go_instances, ic_fan).

    ic_fan = mean over proteins of that protein's own mean GO-IC -- computed
    directly from the per-protein annotation, not reverse-engineered from the
    aggregate matrix (see module docstring).
    """
    opener = gzip.open if path.suffix == ".gz" else open
    go_counter = Counter()
    n_proteins = 0
    total_instances = 0
    per_protein_ics = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            gos = [g.strip() for g in parts[1].split(",") if g.strip()]
            if not gos:
                continue
            n_proteins += 1
            total_instances += len(gos)
            for g in gos:
                go_counter[g] += 1
            ics = [ic_map[g] for g in gos if g in ic_map]
            if ics:
                per_protein_ics.append(sum(ics) / len(ics))
    ic_fan = float(np.mean(per_protein_ics)) if per_protein_ics else np.nan
    return go_counter, n_proteins, total_instances, ic_fan


def stats_row_from_counts(n_proteins: int, total_instances: int, n_unique_go: int, ic_fan: float) -> dict:
    return {
        "Total_prots": n_proteins,
        "Total_mRNA": np.nan,
        "Total_fan": total_instances,
        "Total_hom": np.nan,
        "Unique_fan": n_unique_go,
        "Unique_hom": np.nan,
        "Perc_fan": 1.0 if n_proteins > 0 else np.nan,
        "Perc_hom": np.nan,
        "GO/Prot_fan": (total_instances / n_proteins) if n_proteins > 0 else np.nan,
        "GO/Prot_hom": np.nan,
        "IC_fan": ic_fan,
        "IC_hom": np.nan,
    }


# --------------------------------------------------------- species discovery
def discover_asgard(topgo_dir: Path, metadata_path: Path):
    meta = pd.read_csv(metadata_path, sep="\t")
    id_to_clade = dict(zip(meta["ID"], meta["Group"]))
    entries = []
    for f in sorted(topgo_dir.glob("*_topgo.txt")):
        code = f.name[: -len("_topgo.txt")]
        clade = id_to_clade.get(code)
        if clade is None:
            _log(f"  [WARN] asgard: {code} no está en {metadata_path.name}, se omite")
            continue
        entries.append({"code": code, "species": code, "group": "Asgard", "path": f, "_clade": clade})
    return entries


def discover_metazoa(topgo_dir: Path, metadata_path: Path):
    meta = pd.read_csv(metadata_path, sep="\t")
    id_to_name = dict(zip(meta["ID"], meta["SCIENTIFIC_NAME"]))
    entries = []
    for f in sorted(topgo_dir.glob("*_fantasia_topgo.txt*")):
        name_part = f.name
        for suf in ("_fantasia_topgo.txt.gz", "_fantasia_topgo.txt"):
            if name_part.endswith(suf):
                code = name_part[: -len(suf)]
                break
        else:
            continue
        sci_name = id_to_name.get(code)
        if sci_name is None:
            _log(f"  [WARN] metazoa: {code} no está en {metadata_path.name}, se omite")
            continue
        species = sci_name.strip().replace(" ", "_")
        entries.append({"code": code, "species": species, "group": "Metazoa", "path": f})
    return entries


def discover_protists(topgo_dir: Path, metadata_path: Path):
    meta = pd.read_csv(metadata_path, sep="\t")
    seen_species = {}
    entries = []
    fallback_warned = set()
    files = {f.name[: -len("_topgo.txt")]: f for f in topgo_dir.glob("*_topgo.txt")}
    for _, row in meta.iterrows():
        code = row["AN"]
        species = str(row["SP ID"]).strip()
        if species in seen_species:
            _log(f"  [WARN] protists: especie duplicada '{species}' ({code} y {seen_species[species]}), se mantiene la primera")
            continue
        seen_species[species] = code
        f = files.get(code)
        if f is None:
            _log(f"  [WARN] protists: {code} ({species}) no tiene topgo en {topgo_dir}, se omite")
            continue
        supergroup = row.get("supergroup")
        if supergroup == "Archaeplastida":
            group = row.get("division")
        elif supergroup == "Obazoa":
            group = row.get("subdivision")
        else:
            group = row.get("division")
            if supergroup not in fallback_warned:
                _log(f"  [WARN] protists: supergroup '{supergroup}' sin regla definida, usando 'division' ({group}) — revisar")
                fallback_warned.add(supergroup)
        if pd.isna(group):
            _log(f"  [WARN] protists: {species} ({code}) sin valor de grupo válido, se omite")
            continue
        entries.append({"code": code, "species": species, "group": str(group), "path": f})
    return entries


# ------------------------------------------------------------- group runner
def process_group(group_name: str, entries: list, ic_map: dict, workdir: Path, force: bool):
    long_path = workdir / f"{group_name}_long.tsv.gz"
    stats_path = workdir / f"{group_name}_stats.tsv"
    taxons_path = workdir / f"{group_name}_taxons.tsv"

    if (_checkpoint(long_path, f"{group_name} counts", force)
            and _checkpoint(stats_path, f"{group_name} stats", force)
            and _checkpoint(taxons_path, f"{group_name} taxons", force)):
        return

    _log(f"  Parsing {len(entries)} topgo files ({group_name}) ...")
    long_rows = []
    stats_rows = []
    taxon_rows = []
    t0 = time.monotonic()
    for i, e in enumerate(entries, 1):
        go_counter, n_proteins, total_instances, ic_fan = parse_topgo_file(e["path"], ic_map)
        if n_proteins == 0:
            _log(f"  [WARN] {e['species']} ({e['code']}): 0 proteínas anotadas, se omite")
            continue
        for go_id, count in go_counter.items():
            long_rows.append((e["species"], go_id, count))
        row = stats_row_from_counts(n_proteins, total_instances, len(go_counter), ic_fan)
        row["Species"] = e["species"]
        stats_rows.append(row)
        taxon_rows.append({"Group": e["group"], "Species": e["species"]})
        if i % 200 == 0:
            _log(f"    ... {i}/{len(entries)} procesados ({time.monotonic()-t0:.0f}s)")

    long_df = pd.DataFrame(long_rows, columns=["Species", "GO", "Count"])
    long_df.to_csv(long_path, sep="\t", index=False, compression="gzip")

    stats_df = pd.DataFrame(stats_rows)[["Species"] + STATS_COLUMNS]
    stats_df.to_csv(stats_path, sep="\t", index=False)

    taxons_df = pd.DataFrame(taxon_rows)[["Group", "Species"]]
    taxons_df.to_csv(taxons_path, sep="\t", index=False)

    _log(f"  {group_name}: {len(stats_rows)} especies con anotación, "
         f"{long_df['GO'].nunique()} GO terms distintos")


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asgard-dir", type=Path, help="Directorio con los topgo de Asgard (topgo_tables_asgard)")
    ap.add_argument("--asgard-metadata", type=Path, help="metadata_asgard.txt (columnas ID, Group)")
    ap.add_argument("--metazoa-dir", type=Path, help="Directorio con los topgo de Metazoa (topgo_tables_metazoa)")
    ap.add_argument("--metazoa-metadata", type=Path, help="metadata_metazoa.txt (columnas ID, SCIENTIFIC_NAME, ID_NCBI, Phylum)")
    ap.add_argument("--protists-dir", type=Path, help="Directorio con los topgo de Protists (topgo_tables_protists)")
    ap.add_argument("--protists-metadata", type=Path, help="metadata_protists.txt (columnas AN, SP ID, ..., supergroup, division, subdivision)")

    ap.add_argument("--current-matrix", type=Path, default=Path("merged_PCA_fantasia.tsv"))
    ap.add_argument("--current-stats", type=Path, default=Path("merged_species_stats.tsv"))
    ap.add_argument("--current-taxons", type=Path, default=Path("merged_taxons.tsv"))
    ap.add_argument("--ic-file", type=Path, default=DEFAULT_IC_PATH, help="GO id -> IC TSV (default: bundled data/All_GOs_ic.tsv)")

    ap.add_argument("--matrix-out", type=Path, default=None, help="Default: overwrite --current-matrix (with a timestamped backup)")
    ap.add_argument("--stats-out", type=Path, default=None, help="Default: overwrite --current-stats (with a timestamped backup)")
    ap.add_argument("--taxons-out", type=Path, default=None, help="Default: overwrite --current-taxons (with a timestamped backup)")

    ap.add_argument("--output", default="build_new_species_run", help="Run directory for logs/workdir/results")

    ap.add_argument("--skip_asgard", action="store_true", help="Skip Module 1 — Asgard archaea ingestion")
    ap.add_argument("--skip_metazoa", action="store_true", help="Skip Module 2 — Metazoa ingestion")
    ap.add_argument("--skip_protists", action="store_true", help="Skip Module 3 — Protists ingestion")
    ap.add_argument("--skip_merge", action="store_true", help="Parse/checkpoint per-group tables only; do not write the merged matrix/stats/taxons")

    ap.add_argument("--force", action="store_true", help="Rerun all steps from scratch even if intermediate outputs exist in workdir/")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--disable_co2_tracking", action="store_true", help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def _validate_inputs(pairs):
    ok = True
    for flag, path in pairs:
        if path is not None and not path.exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


def main():
    args = parse_args()

    for attr in ("asgard_dir", "asgard_metadata", "metazoa_dir", "metazoa_metadata",
                 "protists_dir", "protists_metadata", "current_matrix", "current_stats",
                 "current_taxons", "ic_file"):
        val = getattr(args, attr)
        if val is not None:
            setattr(args, attr, val.resolve())

    checks = []
    if not args.skip_asgard:
        checks += [("--asgard-dir", args.asgard_dir), ("--asgard-metadata", args.asgard_metadata)]
    if not args.skip_metazoa:
        checks += [("--metazoa-dir", args.metazoa_dir), ("--metazoa-metadata", args.metazoa_metadata)]
    if not args.skip_protists:
        checks += [("--protists-dir", args.protists_dir), ("--protists-metadata", args.protists_metadata)]
    for flag, path in checks:
        if path is None:
            print(f"ERROR: {flag} is required (unless the corresponding --skip_* is set)", file=sys.stderr)
            sys.exit(1)
    _validate_inputs(checks)

    run_dir = Path(args.output)
    results = run_dir / "results"
    workdir = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = logs_dir / "Run_MergeFantasiaSpecies.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  MergeFantasiaSpecies {VERSION}  —  Run Log\n{sep}\n")
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

    existing_flags = [args.current_matrix.exists(), args.current_stats.exists(), args.current_taxons.exists()]
    if all(existing_flags):
        has_existing_data = True
    elif not any(existing_flags):
        has_existing_data = False
        _log(f"  No se encontraron --current-matrix/--current-stats/--current-taxons "
             f"({args.current_matrix}, {args.current_stats}, {args.current_taxons}) — "
             f"se generarán tablas NUEVAS solo con las especies de este batch (sin fusionar con nada previo)")
    else:
        print("ERROR: de --current-matrix/--current-stats/--current-taxons, algunos existen y otros no "
              "— o pasa las tres rutas de un dataset consistente, o ninguna (para generar tablas nuevas desde cero):",
              file=sys.stderr)
        for flag, p, ok in zip(("--current-matrix", "--current-stats", "--current-taxons"), (args.current_matrix, args.current_stats, args.current_taxons), existing_flags):
            print(f"  {flag}: {p} {'(existe)' if ok else '(NO existe)'}", file=sys.stderr)
        sys.exit(1)

    has_ic_file = args.ic_file.exists()
    if not has_ic_file:
        _log(f"  [WARN] --ic-file no encontrado ({args.ic_file}) — se continúa sin IC "
             f"(IC_fan quedará vacío para las especies nuevas)")

    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif workdir.exists() and any(workdir.iterdir()):
        _log("Existing workdir found — resuming from checkpoints (use --force to rerun all steps from scratch)")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Asgard    : {'skip' if args.skip_asgard else args.asgard_dir}")
        _log(f"  Metazoa   : {'skip' if args.skip_metazoa else args.metazoa_dir}")
        _log(f"  Protists  : {'skip' if args.skip_protists else args.protists_dir}")
        _log(f"  Matrix out  : {matrix_out}")
        _log(f"  Stats out   : {stats_out}")
        _log(f"  Taxons out  : {taxons_out}")
        _log(f"  Modo        : {'fusión con datos existentes' if has_existing_data else 'tablas nuevas desde cero (no se encontraron datos previos)'}")
        _log(f"  IC file     : {args.ic_file if has_ic_file else 'no encontrado — IC_fan quedará vacío'}")
        _log("  Steps that would run:")
        if not args.skip_asgard:
            _log("    [1] Ingest Asgard   → workdir/asgard_{long,stats,taxons}.*")
        if not args.skip_metazoa:
            _log("    [2] Ingest Metazoa  → workdir/metazoa_{long,stats,taxons}.*")
        if not args.skip_protists:
            _log("    [3] Ingest Protists → workdir/protists_{long,stats,taxons}.*")
        if not args.skip_merge:
            _log("    [4] Merge into matrix/stats/taxons → results/ + live files above")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="MergeFantasiaSpecies", log_level="warning")
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
        _log("  Sin archivo IC — IC_fan se dejará vacío (NaN) para todas las especies nuevas")

    groups_processed = []
    if not args.skip_asgard:
        _banner("Módulo 1 — Asgard")
        entries = discover_asgard(args.asgard_dir, args.asgard_metadata)
        _log(f"  {len(entries)} archivos topgo encontrados con metadata válida")
        process_group("asgard", entries, ic_map, workdir, args.force)
        groups_processed.append("asgard")

    if not args.skip_metazoa:
        _banner("Módulo 2 — Metazoa")
        entries = discover_metazoa(args.metazoa_dir, args.metazoa_metadata)
        _log(f"  {len(entries)} archivos topgo encontrados con metadata válida")
        process_group("metazoa", entries, ic_map, workdir, args.force)
        groups_processed.append("metazoa")

    if not args.skip_protists:
        _banner("Módulo 3 — Protists")
        entries = discover_protists(args.protists_dir, args.protists_metadata)
        _log(f"  {len(entries)} archivos topgo encontrados con metadata válida")
        process_group("protists", entries, ic_map, workdir, args.force)
        groups_processed.append("protists")

    n_new_species = 0
    n_new_go = 0
    n_skipped_existing = 0
    if not args.skip_merge:
        _banner("Módulo 4 — Fusión con la matriz/stats/taxons actuales" if has_existing_data
                 else "Módulo 4 — Construcción de tablas nuevas (sin datos previos)")

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
        known_species = set(current_matrix.index) | set(current_taxons["Species"])

        long_parts, stats_parts, taxon_parts = [], [], []
        for g in groups_processed:
            long_parts.append(pd.read_csv(workdir / f"{g}_long.tsv.gz", sep="\t"))
            stats_parts.append(pd.read_csv(workdir / f"{g}_stats.tsv", sep="\t"))
            taxon_parts.append(pd.read_csv(workdir / f"{g}_taxons.tsv", sep="\t"))

        if not long_parts:
            _log("  No hay grupos nuevos que fusionar (todos --skip_*). Nada que hacer.")
        else:
            long_df = pd.concat(long_parts, ignore_index=True)
            stats_df = pd.concat(stats_parts, ignore_index=True)
            taxon_df = pd.concat(taxon_parts, ignore_index=True)

            new_species_mask = ~stats_df["Species"].isin(known_species)
            n_skipped_existing = (~new_species_mask).sum()
            if n_skipped_existing:
                _log(f"  {n_skipped_existing} especies ya presentes en la matriz/taxonomía actual — se omiten")
            new_species = set(stats_df.loc[new_species_mask, "Species"])
            long_df = long_df[long_df["Species"].isin(new_species)]
            stats_df = stats_df[new_species_mask]
            taxon_df = taxon_df[taxon_df["Species"].isin(new_species)]

            _log(f"  {len(new_species)} especies nuevas a añadir")
            n_new_species = len(new_species)

            _log("  Pivotando conteos a formato ancho ...")
            new_wide = long_df.pivot_table(index="Species", columns="GO", values="Count", fill_value=0, aggfunc="sum")
            new_wide = new_wide.astype("int32")
            del long_df
            gc.collect()

            _log(f"  Guardando vista previa solo de las especies nuevas ({new_wide.shape[0]} x {new_wide.shape[1]}) ...")
            new_wide.to_csv(results / f"mod01_new_species_counts_{prefix}.tsv", sep="\t")

            all_go = current_matrix.columns.union(new_wide.columns)
            n_new_go = len(all_go) - len(current_matrix.columns)
            _log(f"  {n_new_go} GO terms nuevos no presentes en la matriz actual")

            current_matrix = current_matrix.reindex(columns=all_go, fill_value=0)
            new_wide = new_wide.reindex(columns=all_go, fill_value=0)
            merged_matrix = pd.concat([current_matrix, new_wide], axis=0).astype("int32")
            del current_matrix, new_wide
            gc.collect()

            merged_stats = pd.concat([current_stats, stats_df[["Species"] + STATS_COLUMNS]], ignore_index=True)
            merged_taxons = pd.concat([current_taxons, taxon_df[["Group", "Species"]]], ignore_index=True)

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
            merged_stats.to_csv(results / f"mod02_stats_{prefix}.tsv", sep="\t", index=False)

            _log(f"  Escribiendo taxonomía fusionada ({len(merged_taxons)} especies) → {taxons_out}")
            merged_taxons.to_csv(taxons_out, sep="\t", index=False)
            merged_taxons.to_csv(results / f"mod03_taxons_{prefix}.tsv", sep="\t", index=False)

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
        "groups_processed": groups_processed,
        "n_new_species": int(n_new_species),
        "n_new_go_terms": int(n_new_go),
        "n_skipped_already_present": int(n_skipped_existing),
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
    _log(f"  Tiempo total             : {elapsed_s:.1f}s")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
