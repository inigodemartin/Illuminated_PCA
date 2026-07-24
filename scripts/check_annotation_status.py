#!/usr/bin/env python3
"""
Read-only monitoring script -- does not write to any of the input files.

Scans a non_viridiplantae_species base dir and a viridiplantae_species base
dir (same per-species layout as fungi_structure.txt:
{Species}/04_FunctionalAnnotation/FANTASIA_2025_*/*_GOs_merged.tsv and
Homology_annot_*/*.proteins.funct_ahrd.tsv), and optionally a third source
for species someone else already ran (--belen-dir, layout as in
belen_species_tree.txt: flat topgo_tables_{asgard,metazoa,protists}/ dirs,
one file per species code, no homology step at all -- species identity is
resolved via --metadata-{asgard,metazoa,protists}). Cross-checks what it
finds against the species roster (--taxons) and the merged GO-count matrix
(--matrix), to answer:

  - How many species ran successfully overall, and per taxonomic group?
  - Of those, how many ran WITH homology annotation (AHRD) -- only
    meaningful for the non_viridiplantae/viridiplantae source, since the
    belen_* source never had a homology step?
  - Of those, how many made it into the matrix?
  - Any species on disk that aren't in the roster at all?
  - Any species in the roster with no directory on disk at all (never ran)?
"""

VERSION = "v0.1.0"

import argparse
import getpass
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

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


# --------------------------------------------------------- species discovery
def discover_species(base_dir: Path, label: str) -> list:
    """One row per species subdirectory of base_dir: whether FANTASIA produced
    a non-empty *_GOs_merged.tsv (Ran_ok) and whether homology annotation
    produced a non-empty *.proteins.funct_ahrd.tsv (Has_homology)."""
    entries = []
    for sp_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        species = sp_dir.name
        fa_dir = sp_dir / "04_FunctionalAnnotation"
        fantasia_matches = sorted(fa_dir.glob("FANTASIA_2025_*/*_GOs_merged.tsv")) if fa_dir.is_dir() else []
        ahrd_matches = sorted(fa_dir.glob("Homology_annot_*/*.proteins.funct_ahrd.tsv")) if fa_dir.is_dir() else []
        ran_ok = any(p.stat().st_size > 0 for p in fantasia_matches)
        has_homology = any(p.stat().st_size > 0 for p in ahrd_matches)
        entries.append({
            "Species": species,
            "Source_dir": label,
            "Ran_ok": ran_ok,
            "Has_homology": has_homology,
            "Homology_tracked": True,
        })
    return entries


SCAN_COLUMNS = ["Species", "Source_dir", "Ran_ok", "Has_homology", "Homology_tracked"]


def run_scan(base_dir: Path, label: str, workdir: Path, force: bool) -> pd.DataFrame:
    out_path = workdir / f"{label}_scan.tsv"
    if _checkpoint(out_path, f"scan {label}", force):
        return pd.read_csv(out_path, sep="\t")
    _log(f"  Escaneando {base_dir} ({label}) ...")
    entries = discover_species(base_dir, label)
    df = pd.DataFrame(entries, columns=SCAN_COLUMNS)
    df.to_csv(out_path, sep="\t", index=False)
    _log(f"  {len(df)} directorios de especie encontrados en {label} "
         f"({int(df['Ran_ok'].sum())} con salida FANTASIA no vacía)")
    return df


# ---------------------------------------------- "belen" species discovery
# Someone else already ran FANTASIA for these species with no per-species
# directory and no homology step -- just one flat topgo-table file per
# species code under --belen-dir, and a separate metadata file mapping each
# code to the species name used in --taxons/--matrix (see
# belen_species_tree.txt for the on-disk layout this expects).
BELEN_GROUPS = [
    {"label": "asgard", "subdir": "topgo_tables_asgard", "suffix": "_topgo.txt", "id_col": "ID", "species_col": "ID"},
    {"label": "metazoa", "subdir": "topgo_tables_metazoa", "suffix": "_fantasia_topgo.txt.gz", "id_col": "ID", "species_col": "SCIENTIFIC_NAME"},
    {"label": "protists", "subdir": "topgo_tables_protists", "suffix": "_topgo.txt", "id_col": "AN", "species_col": "SP ID"},
]


def discover_belen_group(topgo_dir: Path, suffix: str, metadata_df: pd.DataFrame, id_col: str, species_col: str, label: str):
    """Returns (matched_entries, orphan_codes). matched_entries have a Species
    name resolved via metadata and feed into the main scan; orphan_codes are
    topgo-table files on disk whose code has no metadata row, so no Species
    name can be resolved for them -- reported separately instead."""
    file_sizes = {}
    if topgo_dir.is_dir():
        for f in topgo_dir.iterdir():
            if f.name.endswith(suffix):
                file_sizes[f.name[:-len(suffix)]] = f.stat().st_size

    # selecting [id_col, species_col] when they're the same name (asgard: code
    # IS the species name) would give two identically-named columns, and
    # row[id_col] would then return a Series instead of a scalar.
    cols = [id_col] if id_col == species_col else [id_col, species_col]
    meta = metadata_df[cols].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    matched = []
    seen_codes = set()
    for _, row in meta.iterrows():
        code = str(row[id_col]).strip()
        species = str(row[species_col]).strip().replace(" ", "_")
        seen_codes.add(code)
        size = file_sizes.get(code)
        matched.append({
            "Species": species,
            "Source_dir": f"belen_{label}",
            "Ran_ok": size is not None and size > 0,
            "Has_homology": False,
            "Homology_tracked": False,
        })

    orphans = [{"Group": label, "Code": code, "File_size_bytes": size}
               for code, size in sorted(file_sizes.items()) if code not in seen_codes]
    return matched, orphans


def run_belen_scan(belen_dir: Path, metadata_paths: dict, workdir: Path, results: Path, prefix: str, force: bool) -> pd.DataFrame:
    out_path = workdir / "belen_scan.tsv"
    orphans_path = results / f"belen_codes_without_metadata_{prefix}.tsv"
    if _checkpoint(out_path, "scan belen", force):
        return pd.read_csv(out_path, sep="\t")

    all_matched, all_orphans = [], []
    for cfg in BELEN_GROUPS:
        topgo_dir = belen_dir / cfg["subdir"]
        if not topgo_dir.is_dir():
            _log(f"  [WARN] {topgo_dir} no existe — se omite el grupo belen_{cfg['label']}")
            continue
        metadata_df = pd.read_csv(metadata_paths[cfg["label"]], sep="\t")
        matched, orphans = discover_belen_group(topgo_dir, cfg["suffix"], metadata_df, cfg["id_col"], cfg["species_col"], cfg["label"])
        n_ran = sum(1 for e in matched if e["Ran_ok"])
        _log(f"  belen_{cfg['label']}: {len(matched)} códigos en metadata, {n_ran} con topgo table no vacía, "
             f"{len(orphans)} ficheros en disco sin entrada en metadata")
        all_matched.extend(matched)
        all_orphans.extend(orphans)

    if all_orphans:
        pd.DataFrame(all_orphans).to_csv(orphans_path, sep="\t", index=False)
        _log(f"  {len(all_orphans)} ficheros topgo en total sin entrada en metadata — ver {orphans_path.name}")

    df = pd.DataFrame(all_matched, columns=SCAN_COLUMNS)
    df.to_csv(out_path, sep="\t", index=False)
    return df


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--non-viridiplantae-dir", type=Path, required=True,
                     help="non_viridiplantae_species base dir (one subdirectory per species)")
    ap.add_argument("--viridiplantae-dir", type=Path, required=True,
                     help="viridiplantae_species base dir (one subdirectory per species)")
    ap.add_argument("--taxons", type=Path, required=True,
                     help="Species roster with Group column (e.g. merged_taxons_belen.tsv) -- "
                          "the source of truth for the total species count per group")
    ap.add_argument("--matrix", type=Path, required=True,
                     help="Merged GO-count matrix (e.g. merged_PCA_belen_fantasia.tsv) -- "
                          "only its Species column is read")
    ap.add_argument("--belen-dir", type=Path, default=None,
                     help="Optional: base dir with topgo_tables_{asgard,metazoa,protists}/ (layout as in "
                          "belen_species_tree.txt) for species run by someone else, with no homology step. "
                          "Requires --metadata-asgard/--metadata-metazoa/--metadata-protists too.")
    ap.add_argument("--metadata-asgard", type=Path, default=None, help="ID -> Group metadata TSV for the Asgard belen species")
    ap.add_argument("--metadata-metazoa", type=Path, default=None, help="ID -> SCIENTIFIC_NAME metadata TSV for the Metazoa belen species")
    ap.add_argument("--metadata-protists", type=Path, default=None, help="AN -> SP ID metadata TSV for the Protists belen species")
    ap.add_argument("--output", default="check_annotation_status_run", help="Run directory for logs/workdir/results")
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
    args.taxons = args.taxons.resolve()
    args.matrix = args.matrix.resolve()

    validation_pairs = [
        ("--non-viridiplantae-dir", args.non_viridiplantae_dir),
        ("--viridiplantae-dir", args.viridiplantae_dir),
        ("--taxons", args.taxons),
        ("--matrix", args.matrix),
    ]

    belen_metadata_flags = {
        "asgard": "--metadata-asgard", "metazoa": "--metadata-metazoa", "protists": "--metadata-protists",
    }
    if args.belen_dir is not None:
        args.belen_dir = args.belen_dir.resolve()
        missing = [flag for label, flag in belen_metadata_flags.items() if getattr(args, f"metadata_{label}") is None]
        if missing:
            print(f"ERROR: --belen-dir requires {', '.join(belen_metadata_flags.values())} to all be given "
                  f"(missing: {', '.join(missing)})", file=sys.stderr)
            sys.exit(1)
        for label, flag in belen_metadata_flags.items():
            path = getattr(args, f"metadata_{label}").resolve()
            setattr(args, f"metadata_{label}", path)
            validation_pairs.append((flag, path))
        validation_pairs.append(("--belen-dir", args.belen_dir))

    _validate_inputs(validation_pairs)

    run_dir = Path(args.output)
    results = run_dir / "results"
    workdir = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = logs_dir / "Run_CheckAnnotationStatus.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  CheckAnnotationStatus {VERSION}  —  Run Log\n{sep}\n")
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
        _log(f"  Taxons (roster)       : {args.taxons}")
        _log(f"  Matrix                : {args.matrix}")
        _log(f"  Belen dir             : {args.belen_dir if args.belen_dir else '(no indicado — se omite)'}")
        _log(f"  Output                : {run_dir}/")
        _log("  Steps that would run:")
        _log("    [1] Escanear non_viridiplantae_dir → workdir/non_viridiplantae_scan.tsv")
        _log("    [2] Escanear viridiplantae_dir     → workdir/viridiplantae_scan.tsv")
        if args.belen_dir:
            _log("    [2b] Escanear belen_dir (asgard/metazoa/protists) → workdir/belen_scan.tsv")
        _log("    [3] Cruzar contra taxons y matrix   → results/*.tsv")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="CheckAnnotationStatus", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped (conda install -c conda-forge codecarbon)")

    t_start = time.monotonic()

    _banner("Módulo 1 — Escaneo de directorios de especies")
    scan_nv = run_scan(args.non_viridiplantae_dir, "non_viridiplantae", workdir, args.force)
    scan_vp = run_scan(args.viridiplantae_dir, "viridiplantae", workdir, args.force)
    scan_parts = [scan_nv, scan_vp]

    if args.belen_dir is not None:
        _banner("Módulo 1b — Especies belen (Asgard/Metazoa/Protists)")
        metadata_paths = {"asgard": args.metadata_asgard, "metazoa": args.metadata_metazoa, "protists": args.metadata_protists}
        scan_parts.append(run_belen_scan(args.belen_dir, metadata_paths, workdir, results, prefix, args.force))
    else:
        _log("  --belen-dir no indicado — se omite el chequeo de especies Asgard/Metazoa/Protists")

    scan_df = pd.concat(scan_parts, ignore_index=True)

    dup_dirs = scan_df[scan_df.duplicated(subset="Species", keep=False)].sort_values("Species")
    if len(dup_dirs):
        n_dup_species = dup_dirs["Species"].nunique()
        _log(f"  [WARN] {n_dup_species} especies aparecen en MÁS DE UNA fuente "
             f"(non_viridiplantae/viridiplantae/belen_*) — ver duplicate_species_dirs_{prefix}.tsv")
        dup_dirs.to_csv(results / f"duplicate_species_dirs_{prefix}.tsv", sep="\t", index=False)

    _banner("Módulo 2 — Cruce con taxonomía (roster) y matriz")
    _log(f"  Cargando roster ({args.taxons.name}) ...")
    taxons_raw = pd.read_csv(args.taxons, sep="\t")
    n_taxons_dup_rows = int(taxons_raw["Species"].duplicated().sum())
    if n_taxons_dup_rows:
        _log(f"  [WARN] {n_taxons_dup_rows} filas duplicadas de Species en {args.taxons.name} "
             f"(se usa la primera aparición de cada una para el conteo por grupo)")
    taxons_dedup = taxons_raw.drop_duplicates(subset="Species", keep="first")
    taxons_map = dict(zip(taxons_dedup["Species"], taxons_dedup["Group"]))
    taxons_species = set(taxons_map.keys())

    _log(f"  Cargando matriz ({args.matrix.name}) ...")
    matrix_species = set(pd.read_csv(args.matrix, sep="\t", usecols=["Species"])["Species"])

    scan_grouped = scan_df.groupby("Species").agg(
        Found_dirs=("Source_dir", lambda x: ",".join(sorted(set(x)))),
        Ran_ok=("Ran_ok", "any"),
        Has_homology=("Has_homology", "any"),
        Homology_tracked=("Homology_tracked", "any"),
    ).reset_index()

    all_species = sorted(set(scan_grouped["Species"]) | taxons_species | matrix_species)
    master = pd.DataFrame({"Species": all_species})
    master["In_taxons"] = master["Species"].isin(taxons_species)
    master["Group"] = master["Species"].map(taxons_map).fillna("UNLISTED")
    master = master.merge(scan_grouped, on="Species", how="left")
    master["Found_dirs"] = master["Found_dirs"].fillna("")
    # merge() upcasts bool columns to object when NaN is introduced for
    # unmatched rows; fillna(False) alone leaves them as Python bools stored
    # in an object-dtype column, where "~" does bitwise-not on the *object*
    # (~True == -2, ~False == -1 -- both truthy!) instead of logical negation.
    master["Ran_ok"] = master["Ran_ok"].fillna(False).astype(bool)
    master["Has_homology"] = master["Has_homology"].fillna(False).astype(bool)
    master["Homology_tracked"] = master["Homology_tracked"].fillna(False).astype(bool)
    master["In_matrix"] = master["Species"].isin(matrix_species)
    master = master.sort_values(["Group", "Species"]).reset_index(drop=True)
    master.to_csv(results / f"species_status_{prefix}.tsv", sep="\t", index=False)

    group_rows = []
    for g in sorted(master["Group"].unique()):
        sub = master[master["Group"] == g]
        total_taxons = int((taxons_dedup["Group"] == g).sum())
        ran_ok = int(sub["Ran_ok"].sum())
        with_hom = int((sub["Ran_ok"] & sub["Has_homology"] & sub["Homology_tracked"]).sum())
        without_hom = int((sub["Ran_ok"] & ~sub["Has_homology"] & sub["Homology_tracked"]).sum())
        hom_not_tracked = int((sub["Ran_ok"] & ~sub["Homology_tracked"]).sum())
        in_matrix = int((sub["Ran_ok"] & sub["In_matrix"]).sum())
        ran_not_in_matrix = ran_ok - in_matrix
        never_ran = int((sub["In_taxons"] & ~sub["Ran_ok"] & (sub["Found_dirs"] == "")).sum())
        dir_no_output = int(((sub["Found_dirs"] != "") & ~sub["Ran_ok"]).sum())
        group_rows.append({
            "Group": g,
            "N_total_taxons": total_taxons,
            "N_ran_ok": ran_ok,
            "N_ran_with_homology": with_hom,
            "N_ran_without_homology": without_hom,
            "N_ran_homology_not_tracked": hom_not_tracked,
            "N_in_matrix": in_matrix,
            "N_ran_not_in_matrix": ran_not_in_matrix,
            "N_never_ran_no_dir": never_ran,
            "N_dir_but_no_output": dir_no_output,
        })
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(results / f"group_summary_{prefix}.tsv", sep="\t", index=False)

    unlisted = master[(~master["In_taxons"]) & master["Ran_ok"]]
    unlisted.to_csv(results / f"unlisted_species_{prefix}.tsv", sep="\t", index=False)

    never_ran_df = master[master["In_taxons"] & ~master["Ran_ok"] & (master["Found_dirs"] == "")]
    never_ran_df.to_csv(results / f"never_ran_species_{prefix}.tsv", sep="\t", index=False)

    # Only flag "ran without homology" where homology is actually tracked for
    # the source (non_viridiplantae/viridiplantae) -- for belen_* species
    # (Homology_tracked=False) this was never a step to begin with, so it's
    # not a gap worth flagging the same way.
    no_homology_df = master[master["Ran_ok"] & master["Homology_tracked"] & ~master["Has_homology"]]
    no_homology_df.to_csv(results / f"ran_without_homology_{prefix}.tsv", sep="\t", index=False)

    not_in_matrix_df = master[master["Ran_ok"] & ~master["In_matrix"]]
    not_in_matrix_df.to_csv(results / f"ran_not_in_matrix_{prefix}.tsv", sep="\t", index=False)

    roster_mask = master["Group"] != "UNLISTED"
    n_total_taxons = len(taxons_species)
    n_ran_ok_roster = int((master["Ran_ok"] & roster_mask).sum())
    n_ran_with_hom_roster = int((master["Ran_ok"] & master["Has_homology"] & master["Homology_tracked"] & roster_mask).sum())
    n_hom_not_tracked_roster = int((master["Ran_ok"] & ~master["Homology_tracked"] & roster_mask).sum())
    n_in_matrix_roster = int((master["Ran_ok"] & master["In_matrix"] & roster_mask).sum())

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
        "n_total_species_taxons": n_total_taxons,
        "n_ran_ok": n_ran_ok_roster,
        "n_ran_with_homology": n_ran_with_hom_roster,
        "n_ran_without_homology": n_ran_ok_roster - n_ran_with_hom_roster - n_hom_not_tracked_roster,
        "n_ran_homology_not_tracked": n_hom_not_tracked_roster,
        "n_in_matrix": n_in_matrix_roster,
        "n_ran_not_in_matrix": n_ran_ok_roster - n_in_matrix_roster,
        "n_never_ran_no_dir": int(never_ran_df.shape[0]),
        "n_unlisted_on_disk": int(unlisted.shape[0]),
        "n_duplicate_dirs": int(dup_dirs["Species"].nunique()) if len(dup_dirs) else 0,
        "n_duplicate_taxons_rows": n_taxons_dup_rows,
        "parameters": {
            "non_viridiplantae_dir": str(args.non_viridiplantae_dir),
            "viridiplantae_dir": str(args.viridiplantae_dir),
            "taxons": str(args.taxons),
            "matrix": str(args.matrix),
            "belen_dir": str(args.belen_dir) if args.belen_dir else None,
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

    _banner("Resumen por grupo taxonómico")
    header = (f"  {'Group':<16}{'Total':>8}{'Ran_ok':>9}{'+Homol':>9}{'-Homol':>9}{'N/A_Hom':>9}"
              f"{'In_mtx':>9}{'Not_mtx':>9}{'NeverRan':>10}")
    _log(header)
    for row in group_rows:
        _log(f"  {row['Group']:<16}{row['N_total_taxons']:>8}{row['N_ran_ok']:>9}"
             f"{row['N_ran_with_homology']:>9}{row['N_ran_without_homology']:>9}{row['N_ran_homology_not_tracked']:>9}"
             f"{row['N_in_matrix']:>9}{row['N_ran_not_in_matrix']:>9}{row['N_never_ran_no_dir']:>10}")

    _banner("Listo")
    _log(f"  Especies en el roster (taxons)         : {n_total_taxons}")
    _log(f"  Especies corridas OK                    : {n_ran_ok_roster}")
    _log(f"    con homología                         : {n_ran_with_hom_roster}")
    _log(f"    SIN homología                         : {n_ran_ok_roster - n_ran_with_hom_roster - n_hom_not_tracked_roster}")
    _log(f"    homología no aplica (fuente belen_*)  : {n_hom_not_tracked_roster}")
    _log(f"  Corridas OK y presentes en la matriz     : {n_in_matrix_roster}")
    _log(f"  Corridas OK pero AUSENTES de la matriz   : {n_ran_ok_roster - n_in_matrix_roster}")
    _log(f"  En el roster pero sin directorio (nunca corrieron) : {never_ran_df.shape[0]}")
    _log(f"  En disco (corridas) pero NO en el roster : {unlisted.shape[0]}")
    if len(dup_dirs):
        _log(f"  Especies con directorio en ambos base dirs : {dup_dirs['Species'].nunique()}")
    _log(f"  Tiempo total                             : {elapsed_s:.1f}s")
    _log(f"  Resultados en: {results}/")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
