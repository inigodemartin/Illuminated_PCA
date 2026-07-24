#!/usr/bin/env python3
"""
General functional-annotation statistics, built from scratch (does not read
or extend any existing merged_species_stats file).

Scans the same three annotation sources as check_annotation_status.py --
non_viridiplantae_species / viridiplantae_species (FANTASIA GOs_merged.tsv +
optional AHRD homology, per fungi_structure.txt) and, optionally, the belen
species (Asgard/Metazoa/Protists topgo tables + metadata, no homology step,
per belen_species_tree.txt) -- and computes, per species, a wide table of
annotation-quality statistics: proteome coverage, GO richness (overall and
per GO namespace BP/CC/MF), Information Content, and, where homology is
available, agreement/novelty between FANTASIA and homology.

Methodology notes (read before interpreting the output):
  - IC statistics (Mean/Median/SD/Max_IC_*) are computed over the POOLED
    per-(protein, GO)-instance IC values, not a per-protein average of
    per-protein averages -- pooling avoids a protein with one highly
    specific GO term getting the same weight as a protein with many broad
    ones.
  - GO_per_protein_{BP,CC,MF}_* is defined so the three namespace values sum
    exactly to the overall GO_per_protein_* value (same n_annotated
    denominator throughout).
  - Total_prots is the true proteome size (AHRD row count) when homology is
    available; otherwise it falls back to the FANTASIA-annotated protein
    count and Total_prots_is_approx=True is set -- N_unannotated/Pct_*
    columns are not meaningful in that case (they degrade to 0 by
    construction, not because the whole proteome was actually annotated).
"""

VERSION = "v0.1.0"

import argparse
import getpass
import gzip
import json
import os
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from general_pca_common import DEFAULT_IC_PATH, load_go_ic  # noqa: E402

DEFAULT_OBO_PATH = Path(__file__).parent.parent / "data" / "go-basic_2025.obo"

NAMESPACES = ("BP", "CC", "MF")
NAMESPACE_SHORT = {"biological_process": "BP", "cellular_component": "CC", "molecular_function": "MF"}

BELEN_GROUPS = [
    {"label": "asgard", "subdir": "topgo_tables_asgard", "suffix": "_topgo.txt", "id_col": "ID", "species_col": "ID"},
    {"label": "metazoa", "subdir": "topgo_tables_metazoa", "suffix": "_fantasia_topgo.txt.gz", "id_col": "ID", "species_col": "SCIENTIFIC_NAME"},
    {"label": "protists", "subdir": "topgo_tables_protists", "suffix": "_topgo.txt", "id_col": "AN", "species_col": "SP ID"},
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


# --------------------------------------------------------------- GO namespace
def load_go_namespace(obo_file: Path) -> dict:
    """GO id -> "BP"/"CC"/"MF", parsed straight from the OBO (id:/namespace:
    lines only -- no parent/child graph needed here)."""
    ns_map = {}
    current_id = None
    with open(obo_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                current_id = None
            elif line.startswith("id:"):
                current_id = line.split("id:", 1)[1].strip()
            elif line.startswith("namespace:") and current_id is not None:
                ns = line.split("namespace:", 1)[1].strip()
                ns_map[current_id] = NAMESPACE_SHORT.get(ns, ns)
    return ns_map


# ------------------------------------------------------------- file parsing
def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def parse_annotation_file(path: Path, go_col: int, skip_prefixes: tuple, header_prefix, ic_map: dict, ns_map: dict) -> dict:
    """
    Generic protein_id \\t ... \\t comma-separated-GO-list parser (one row
    per protein), shared by FANTASIA/belen topgo tables (go_col=1, no
    header) and AHRD homology tables (go_col=5, header). Collects everything
    needed to summarize this one annotation method for one species:
    protein-level coverage, GO richness, and pooled per-instance IC values,
    split by GO namespace where resolvable.
    """
    protein_ids = set()
    annotated_ids = set()
    go_counter = Counter()
    unique_by_ns = defaultdict(set)
    instances_by_ns = Counter()
    per_protein_counts = []
    ic_values = []
    ic_values_by_ns = defaultdict(list)

    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            if header_prefix is not None and line.startswith(header_prefix):
                continue
            parts = line.split("\t")
            protein_id = parts[0]
            protein_ids.add(protein_id)
            if len(parts) <= go_col:
                continue
            gos = [g.strip() for g in parts[go_col].split(",") if g.strip()]
            if not gos:
                continue
            annotated_ids.add(protein_id)
            per_protein_counts.append(len(gos))
            for g in gos:
                go_counter[g] += 1
                ns = ns_map.get(g)
                if ns:
                    unique_by_ns[ns].add(g)
                    instances_by_ns[ns] += 1
                if g in ic_map:
                    ic = ic_map[g]
                    ic_values.append(ic)
                    if ns:
                        ic_values_by_ns[ns].append(ic)

    return {
        "protein_ids": protein_ids,
        "annotated_ids": annotated_ids,
        "unique_go": set(go_counter.keys()),
        "unique_by_ns": unique_by_ns,
        "total_instances": sum(go_counter.values()),
        "instances_by_ns": instances_by_ns,
        "per_protein_counts": per_protein_counts,
        "ic_values": ic_values,
        "ic_values_by_ns": ic_values_by_ns,
    }


# ------------------------------------------------------------- summarizing
def _dist_stats(values):
    """mean, median, sd (sample, ddof=1), min, max -- all NaN if empty."""
    if not values:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    arr = np.asarray(values, dtype="float64")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(np.median(arr)), sd, float(arr.min()), float(arr.max())


def _method_columns(suffix: str) -> list:
    cols = [
        f"N_annotated_{suffix}", f"Pct_annotated_{suffix}",
        f"Total_GO_instances_{suffix}", f"Unique_GO_terms_{suffix}",
        f"GO_per_protein_{suffix}", f"Median_GO_per_protein_{suffix}",
        f"SD_GO_per_protein_{suffix}", f"Min_GO_per_protein_{suffix}", f"Max_GO_per_protein_{suffix}",
        f"Mean_IC_{suffix}", f"Median_IC_{suffix}", f"SD_IC_{suffix}", f"Max_IC_{suffix}",
    ]
    for ns in NAMESPACES:
        cols += [f"Unique_GO_terms_{ns}_{suffix}", f"GO_per_protein_{ns}_{suffix}", f"Mean_IC_{ns}_{suffix}"]
    return cols


CROSS_METHOD_COLUMNS = [
    "N_unannotated", "Pct_unannotated",
    "N_annotated_union", "Pct_annotated_union",
    "N_annotated_intersection", "Pct_annotated_intersection",
    "Unique_GO_terms_union", "Unique_GO_terms_intersection",
    "Jaccard_fan_vs_hom", "N_GO_fan_only", "N_GO_hom_only",
]

COLUMN_ORDER = (
    ["Species", "Source", "Has_homology", "Total_prots", "Total_prots_is_approx"]
    + CROSS_METHOD_COLUMNS
    + _method_columns("fan")
    + _method_columns("hom")
)


def summarize_method(parsed: dict, suffix: str, total_prots) -> dict:
    row = {}
    n_annotated = len(parsed["annotated_ids"])
    row[f"N_annotated_{suffix}"] = n_annotated
    row[f"Pct_annotated_{suffix}"] = (n_annotated / total_prots) if total_prots else np.nan
    row[f"Total_GO_instances_{suffix}"] = parsed["total_instances"]
    row[f"Unique_GO_terms_{suffix}"] = len(parsed["unique_go"])
    mean_g, med_g, sd_g, min_g, max_g = _dist_stats(parsed["per_protein_counts"])
    row[f"GO_per_protein_{suffix}"] = mean_g
    row[f"Median_GO_per_protein_{suffix}"] = med_g
    row[f"SD_GO_per_protein_{suffix}"] = sd_g
    row[f"Min_GO_per_protein_{suffix}"] = min_g
    row[f"Max_GO_per_protein_{suffix}"] = max_g
    mean_ic, med_ic, sd_ic, _, max_ic = _dist_stats(parsed["ic_values"])
    row[f"Mean_IC_{suffix}"] = mean_ic
    row[f"Median_IC_{suffix}"] = med_ic
    row[f"SD_IC_{suffix}"] = sd_ic
    row[f"Max_IC_{suffix}"] = max_ic
    for ns in NAMESPACES:
        row[f"Unique_GO_terms_{ns}_{suffix}"] = len(parsed["unique_by_ns"].get(ns, ()))
        instances_ns = parsed["instances_by_ns"].get(ns, 0)
        row[f"GO_per_protein_{ns}_{suffix}"] = (instances_ns / n_annotated) if n_annotated else np.nan
        ns_ic = parsed["ic_values_by_ns"].get(ns, [])
        row[f"Mean_IC_{ns}_{suffix}"] = float(np.mean(ns_ic)) if ns_ic else np.nan
    return row


def process_species(species: str, source: str, fan_parsed: dict, hom_parsed) -> dict:
    n_annotated_fan = len(fan_parsed["annotated_ids"])
    if n_annotated_fan == 0:
        return None

    row = {"Species": species, "Source": source}
    has_homology = hom_parsed is not None
    row["Has_homology"] = has_homology

    if has_homology:
        total_prots = len(hom_parsed["protein_ids"])
        row["Total_prots_is_approx"] = False
    else:
        total_prots = n_annotated_fan
        row["Total_prots_is_approx"] = True
    row["Total_prots"] = total_prots

    row.update(summarize_method(fan_parsed, "fan", total_prots))

    if has_homology:
        row.update(summarize_method(hom_parsed, "hom", total_prots))

        union_ids = fan_parsed["annotated_ids"] | hom_parsed["annotated_ids"]
        inter_ids = fan_parsed["annotated_ids"] & hom_parsed["annotated_ids"]
        row["N_annotated_union"] = len(union_ids)
        row["N_annotated_intersection"] = len(inter_ids)
        row["Pct_annotated_union"] = len(union_ids) / total_prots if total_prots else np.nan
        row["Pct_annotated_intersection"] = len(inter_ids) / total_prots if total_prots else np.nan

        fan_go, hom_go = fan_parsed["unique_go"], hom_parsed["unique_go"]
        go_union, go_inter = fan_go | hom_go, fan_go & hom_go
        row["Unique_GO_terms_union"] = len(go_union)
        row["Unique_GO_terms_intersection"] = len(go_inter)
        row["Jaccard_fan_vs_hom"] = (len(go_inter) / len(go_union)) if go_union else np.nan
        row["N_GO_fan_only"] = len(fan_go - hom_go)
        row["N_GO_hom_only"] = len(hom_go - fan_go)

        row["N_unannotated"] = total_prots - len(union_ids)
    else:
        row["N_unannotated"] = max(total_prots - n_annotated_fan, 0)

    row["Pct_unannotated"] = row["N_unannotated"] / total_prots if total_prots else np.nan
    return row


# --------------------------------------------------------- species discovery
def discover_dir_species(base_dir: Path, source: str) -> list:
    """Same layout as fungi_structure.txt: {Species}/04_FunctionalAnnotation/
    FANTASIA_2025_*/*_GOs_merged.tsv (+ optional Homology_annot_*/
    *.proteins.funct_ahrd.tsv)."""
    entries = []
    for sp_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        fa_dir = sp_dir / "04_FunctionalAnnotation"
        if not fa_dir.is_dir():
            continue
        fantasia_matches = sorted(fa_dir.glob("FANTASIA_2025_*/*_GOs_merged.tsv"))
        if not fantasia_matches:
            continue
        if len(fantasia_matches) > 1:
            _log(f"  [WARN] {sp_dir.name}: {len(fantasia_matches)} ficheros GOs_merged.tsv, usando el primero")
        ahrd_matches = sorted(fa_dir.glob("Homology_annot_*/*.proteins.funct_ahrd.tsv"))
        if len(ahrd_matches) > 1:
            _log(f"  [WARN] {sp_dir.name}: {len(ahrd_matches)} ficheros funct_ahrd.tsv, usando el primero")
        entries.append({
            "species": sp_dir.name, "source": source,
            "fan_path": fantasia_matches[0],
            "hom_path": ahrd_matches[0] if ahrd_matches else None,
        })
    return entries


def discover_belen_species(belen_dir: Path, metadata_paths: dict) -> list:
    entries = []
    for cfg in BELEN_GROUPS:
        topgo_dir = belen_dir / cfg["subdir"]
        if not topgo_dir.is_dir():
            _log(f"  [WARN] {topgo_dir} no existe — se omite belen_{cfg['label']}")
            continue
        meta_df = pd.read_csv(metadata_paths[cfg["label"]], sep="\t")
        cols = [cfg["id_col"]] if cfg["id_col"] == cfg["species_col"] else [cfg["id_col"], cfg["species_col"]]
        meta = meta_df[cols].dropna(subset=[cfg["id_col"]]).drop_duplicates(subset=[cfg["id_col"]])
        n_found = 0
        for _, row in meta.iterrows():
            code = str(row[cfg["id_col"]]).strip()
            species = str(row[cfg["species_col"]]).strip().replace(" ", "_")
            path = topgo_dir / f"{code}{cfg['suffix']}"
            if not path.exists() or path.stat().st_size == 0:
                continue
            entries.append({"species": species, "source": f"belen_{cfg['label']}", "fan_path": path, "hom_path": None})
            n_found += 1
        _log(f"  belen_{cfg['label']}: {n_found} / {len(meta)} códigos con topgo table no vacía")
    return entries


def process_source(entries: list, ic_map: dict, ns_map: dict, workdir: Path, label: str, force: bool) -> pd.DataFrame:
    out_path = workdir / f"{label}_annotation_stats.tsv"
    if _checkpoint(out_path, f"{label} stats", force):
        return pd.read_csv(out_path, sep="\t")

    _log(f"  Parsing {len(entries)} especies ({label}) ...")
    rows = []
    n_skipped = 0
    t0 = time.monotonic()
    for i, e in enumerate(entries, 1):
        fan_parsed = parse_annotation_file(e["fan_path"], go_col=1, skip_prefixes=(), header_prefix=None,
                                            ic_map=ic_map, ns_map=ns_map)
        hom_parsed = None
        if e["hom_path"] is not None:
            hom_parsed = parse_annotation_file(e["hom_path"], go_col=5, skip_prefixes=("#",),
                                                header_prefix="Protein-Accession\t", ic_map=ic_map, ns_map=ns_map)
        row = process_species(e["species"], e["source"], fan_parsed, hom_parsed)
        if row is None:
            _log(f"  [WARN] {e['species']}: 0 proteínas anotadas, se omite")
            n_skipped += 1
        else:
            rows.append(row)
        if i % 200 == 0:
            _log(f"    ... {i}/{len(entries)} procesados ({time.monotonic() - t0:.0f}s)")

    df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    df.to_csv(out_path, sep="\t", index=False)
    _log(f"  {label}: {len(df)} especies con estadísticas ({n_skipped} omitidas, {time.monotonic() - t0:.0f}s)")
    return df


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--non-viridiplantae-dir", type=Path, required=True,
                     help="non_viridiplantae_species base dir (one subdirectory per species)")
    ap.add_argument("--viridiplantae-dir", type=Path, required=True,
                     help="viridiplantae_species base dir (one subdirectory per species)")
    ap.add_argument("--belen-dir", type=Path, default=None,
                     help="Optional: base dir with topgo_tables_{asgard,metazoa,protists}/ (layout as in "
                          "belen_species_tree.txt). Requires --metadata-asgard/--metadata-metazoa/--metadata-protists too.")
    ap.add_argument("--metadata-asgard", type=Path, default=None)
    ap.add_argument("--metadata-metazoa", type=Path, default=None)
    ap.add_argument("--metadata-protists", type=Path, default=None)
    ap.add_argument("--taxons", type=Path, default=None,
                     help="Optional: Species/Group roster TSV -- if given, adds a Group column")
    ap.add_argument("--ic-file", type=Path, default=DEFAULT_IC_PATH, help="GO id -> IC TSV (default: bundled data/All_GOs_ic.tsv)")
    ap.add_argument("--obo-file", type=Path, default=DEFAULT_OBO_PATH,
                     help="GO OBO file, for the BP/CC/MF namespace breakdown (default: bundled data/go-basic_2025.obo)")
    ap.add_argument("--output", default="general_annotation_stats_run", help="Run directory for logs/workdir/results")
    ap.add_argument("--force", action="store_true", help="Rerun all steps from scratch even if intermediate outputs exist in workdir/")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--disable_co2_tracking", action="store_true", help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def _validate_inputs(pairs: list) -> None:
    ok = True
    for flag, path in pairs:
        if not path.exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


def main():
    args = parse_args()

    args.non_viridiplantae_dir = args.non_viridiplantae_dir.resolve()
    args.viridiplantae_dir = args.viridiplantae_dir.resolve()

    validation_pairs = [
        ("--non-viridiplantae-dir", args.non_viridiplantae_dir),
        ("--viridiplantae-dir", args.viridiplantae_dir),
    ]

    belen_flags = {"asgard": "--metadata-asgard", "metazoa": "--metadata-metazoa", "protists": "--metadata-protists"}
    if args.belen_dir is not None:
        args.belen_dir = args.belen_dir.resolve()
        missing = [flag for label, flag in belen_flags.items() if getattr(args, f"metadata_{label}") is None]
        if missing:
            print(f"ERROR: --belen-dir requires {', '.join(belen_flags.values())} to all be given "
                  f"(missing: {', '.join(missing)})", file=sys.stderr)
            sys.exit(1)
        for label, flag in belen_flags.items():
            path = getattr(args, f"metadata_{label}").resolve()
            setattr(args, f"metadata_{label}", path)
            validation_pairs.append((flag, path))
        validation_pairs.append(("--belen-dir", args.belen_dir))

    if args.taxons is not None:
        args.taxons = args.taxons.resolve()
        validation_pairs.append(("--taxons", args.taxons))

    _validate_inputs(validation_pairs)

    args.ic_file = args.ic_file.resolve()
    args.obo_file = args.obo_file.resolve()
    has_ic_file = args.ic_file.exists()
    has_obo_file = args.obo_file.exists()

    run_dir = Path(args.output)
    results = run_dir / "results"
    workdir = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = logs_dir / "Run_GeneralAnnotationStats.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  GeneralAnnotationStats {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif workdir.exists() and any(workdir.iterdir()):
        _log("Existing workdir found — resuming from checkpoints (use --force to rerun all steps from scratch)")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Non-viridiplantae dir : {args.non_viridiplantae_dir}")
        _log(f"  Viridiplantae dir     : {args.viridiplantae_dir}")
        _log(f"  Belen dir             : {args.belen_dir if args.belen_dir else '(no indicado — se omite)'}")
        _log(f"  Taxons (Group)        : {args.taxons if args.taxons else '(no indicado — sin columna Group)'}")
        _log(f"  IC file               : {args.ic_file if has_ic_file else 'no encontrado — columnas IC quedarán vacías'}")
        _log(f"  OBO file              : {args.obo_file if has_obo_file else 'no encontrado — desglose BP/CC/MF quedará vacío'}")
        _log(f"  Output                : {run_dir}/")
        _log("  Steps that would run:")
        _log("    [1] Escanear + parsear non_viridiplantae_dir → workdir/non_viridiplantae_annotation_stats.tsv")
        _log("    [2] Escanear + parsear viridiplantae_dir     → workdir/viridiplantae_annotation_stats.tsv")
        if args.belen_dir:
            _log("    [3] Escanear + parsear belen_dir              → workdir/belen_annotation_stats.tsv")
        _log("    [4] Combinar todo → results/mod01_annotation_stats_<prefix>.tsv")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="GeneralAnnotationStats", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped (conda install -c conda-forge codecarbon)")

    t_start = time.monotonic()

    _banner("Cargando IC y namespace GO de referencia")
    if has_ic_file:
        ic_map = load_go_ic(args.ic_file)
        _log(f"  {len(ic_map)} GO ids con IC cargados desde {args.ic_file.name}")
    else:
        ic_map = {}
        _log(f"  [WARN] --ic-file no encontrado ({args.ic_file}) — columnas de IC quedarán vacías")

    if has_obo_file:
        ns_map = load_go_namespace(args.obo_file)
        _log(f"  {len(ns_map)} GO ids con namespace cargados desde {args.obo_file.name}")
    else:
        ns_map = {}
        _log(f"  [WARN] --obo-file no encontrado ({args.obo_file}) — desglose BP/CC/MF quedará vacío")

    _banner("Módulo 1 — non_viridiplantae")
    entries_nv = discover_dir_species(args.non_viridiplantae_dir, "non_viridiplantae")
    _log(f"  {len(entries_nv)} especies con *_GOs_merged.tsv encontrado")
    df_nv = process_source(entries_nv, ic_map, ns_map, workdir, "non_viridiplantae", args.force)

    _banner("Módulo 2 — viridiplantae")
    entries_vp = discover_dir_species(args.viridiplantae_dir, "viridiplantae")
    _log(f"  {len(entries_vp)} especies con *_GOs_merged.tsv encontrado")
    df_vp = process_source(entries_vp, ic_map, ns_map, workdir, "viridiplantae", args.force)

    all_dfs = [df_nv, df_vp]

    if args.belen_dir is not None:
        _banner("Módulo 3 — belen (Asgard/Metazoa/Protists)")
        metadata_paths = {"asgard": args.metadata_asgard, "metazoa": args.metadata_metazoa, "protists": args.metadata_protists}
        entries_belen = discover_belen_species(args.belen_dir, metadata_paths)
        df_belen = process_source(entries_belen, ic_map, ns_map, workdir, "belen", args.force)
        all_dfs.append(df_belen)
    else:
        _log("  --belen-dir no indicado — se omiten especies Asgard/Metazoa/Protists")

    _banner("Módulo 4 — combinando resultados")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.reindex(columns=COLUMN_ORDER)

    if args.taxons is not None:
        taxons_df = pd.read_csv(args.taxons, sep="\t").drop_duplicates(subset="Species")
        group_map = dict(zip(taxons_df["Species"], taxons_df["Group"]))
        combined.insert(1, "Group", combined["Species"].map(group_map))
        n_no_group = int(combined["Group"].isna().sum())
        if n_no_group:
            _log(f"  [WARN] {n_no_group} especies sin Group en {args.taxons.name}")

    combined = combined.sort_values("Species").reset_index(drop=True)
    out_path = results / f"mod01_annotation_stats_{prefix}.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    _log(f"  Escribiendo {len(combined)} especies × {combined.shape[1]} columnas → {out_path}")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass

    n_by_source = combined["Source"].value_counts().to_dict()
    n_with_homology = int(combined["Has_homology"].sum())

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "n_species_total": int(len(combined)),
        "n_species_by_source": {k: int(v) for k, v in n_by_source.items()},
        "n_species_with_homology": n_with_homology,
        "parameters": {
            "non_viridiplantae_dir": str(args.non_viridiplantae_dir),
            "viridiplantae_dir": str(args.viridiplantae_dir),
            "belen_dir": str(args.belen_dir) if args.belen_dir else None,
            "taxons": str(args.taxons) if args.taxons else None,
            "ic_file": str(args.ic_file),
            "obo_file": str(args.obo_file),
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
    _log(f"  Especies totales       : {len(combined)}")
    for src, n in sorted(n_by_source.items()):
        _log(f"    {src:<20}: {n}")
    _log(f"  Con homología          : {n_with_homology} ({n_with_homology / len(combined) * 100:.1f}%)" if len(combined) else "  Con homología          : 0")
    _log(f"  Tiempo total           : {elapsed_s:.1f}s")
    _log(f"  Resultado: {out_path}")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
