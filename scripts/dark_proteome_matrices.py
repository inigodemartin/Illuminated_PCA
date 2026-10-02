#!/usr/bin/env python3
"""
Dark-proteome GO count matrices from paired FANTASIA / homology (AHRD)
annotations.

The "dark proteome" is the fraction of a proteome that classical
homology-based functional annotation leaves without a function. FANTASIA
(embedding-based) does annotate many of these proteins, so for every species
this script compares the two annotation files protein by protein and splits
the FANTASIA-annotated proteins into two groups:

  only_fantasia  annotated by FANTASIA, no GO term from homology (dark)
  both           annotated by FANTASIA and by homology

and builds, over all species, three Species x GO count matrices (same layout
as merged_PCA_belen_fantasia.tsv: one row per species, one column per GO,
integer counts, 0 when absent):

  mod01_fantasia_only_counts_{prefix}.tsv   FANTASIA GO counts, only_fantasia
  mod01_fantasia_both_counts_{prefix}.tsv   FANTASIA GO counts, both
  mod01_homology_all_counts_{prefix}.tsv    homology GO counts, whole proteome

plus a per-species statistics table (mod02) comparing the two groups
(coverage, protein length, GO richness, IC, namespace balance, FANTASIA vs
homology agreement) and a Group/Species table (mod03) for colouring PCAs.

Inputs are the status TSVs produced on the server for each species set
(columns: species, abbr, fantasia_status, fantasia_file, fantasia_size,
homology_status, homology_file, homology_size). Only species with both
statuses == OK are processed; the rest are listed in the log.

"Annotated by homology" means the AHRD Gene-Ontology-Term column is
non-empty: a BLAST hit whose description carries no GO term does NOT count
(decision taken 2026-10-02). The AHRD table lists every protein of the
proteome, so its row count is the proteome size.

Protein lengths come from the proteome FASTA located automatically as
{species_dir}/00_GenomeSource/**/*_5k_removed.pep (the most processed .pep,
the FANTASIA input); when several match, their ID -> length maps are merged.
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
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_IC_PATH = Path(__file__).parent.parent / "data" / "All_GOs_ic.tsv"
DEFAULT_PROTEOME_GLOB = "00_GenomeSource/**/*_5k_removed.pep"

NAMESPACE_SHORT = {"biological_process": "BP", "cellular_component": "CC", "molecular_function": "MF"}
GROUPS = ("only_fantasia", "both")

_LOG_FH = None   # set in main() once logs/ dir exists


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


def _dated_log_path(logs_dir: Path, base_name: str) -> Path:
    """Non-overwriting log path: {base_name}_{YYYYMMDD}.log, or
    {base_name}_{YYYYMMDD}_2.log, _3.log, ... if today's file already
    exists (e.g. multiple runs on the same day)."""
    date_str = datetime.now().strftime("%Y%m%d")
    candidate = logs_dir / f"{base_name}_{date_str}.log"
    if not candidate.exists():
        return candidate
    n = 2
    while (logs_dir / f"{base_name}_{date_str}_{n}.log").exists():
        n += 1
    return logs_dir / f"{base_name}_{date_str}_{n}.log"


def _validate_inputs(pairs: list) -> None:
    """pairs: list of (flag_name, Path)"""
    ok = True
    for flag, path in pairs:
        if not path.exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------ parsers
def load_ic_and_namespace(ic_file: Path):
    """GO -> IC (float) and GO -> BP/CC/MF from the headerless IC TSV
    (go_id, namespace, col3, col4, ic, description)."""
    ic_map, ns_map = {}, {}
    with open(ic_file) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] in ic_map:
                continue
            try:
                ic_map[parts[0]] = float(parts[4])
            except ValueError:
                continue
            ns = NAMESPACE_SHORT.get(parts[1])
            if ns:
                ns_map[parts[0]] = ns
    return ic_map, ns_map


def _split_gos(field: str) -> list:
    return [g.strip() for g in field.split(",") if g.strip()]


def parse_fantasia(path: Path) -> dict:
    """protein_id -> [GO, ...] from a headerless *_GOs_merged.tsv."""
    annots = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gos = _split_gos(parts[1])
            if gos:
                annots[parts[0]] = gos
    return annots


def parse_ahrd(path: Path):
    """All protein IDs (proteome) and protein_id -> [GO, ...] for the rows
    whose Gene-Ontology-Term column (index 5) is non-empty."""
    proteome, annots = [], {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("Protein-Accession"):
                continue
            parts = line.split("\t")
            pid = parts[0]
            if not pid:
                continue
            proteome.append(pid)
            if len(parts) > 5:
                gos = _split_gos(parts[5])
                if gos:
                    annots[pid] = gos
    return proteome, annots


def parse_fasta_lengths(path: Path, lengths: dict) -> None:
    """Add protein_id (first header token) -> length (aa) into `lengths`."""
    pid, n = None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if pid is not None:
                    lengths[pid] = n
                pid = line[1:].split()[0] if len(line) > 1 else None
                n = 0
            else:
                n += len(line.strip().rstrip("*"))
    if pid is not None:
        lengths[pid] = n


def find_proteomes(fantasia_file: Path, pattern: str) -> list:
    """{species_dir}/00_GenomeSource/**/*_5k_removed.pep, where species_dir is
    three levels above the FANTASIA output file
    (species/04_FunctionalAnnotation/FANTASIA_2025*/X_GOs_merged.tsv)."""
    species_dir = fantasia_file.parent.parent.parent
    return sorted(p for p in species_dir.glob(pattern) if p.is_file())


# ------------------------------------------------------------------- stats
def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _median(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _length_pvalue(a: list, b: list) -> float:
    """Two-sided Mann-Whitney U on protein lengths (only_fantasia vs both)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ImportError:
        return float("nan")


def group_stats(ids: list, fan: dict, lengths: dict, ic_map: dict, ns_map: dict, label: str) -> dict:
    """Length, GO richness, IC and namespace statistics for one protein
    group, keyed as {Metric}_{label}. IC is pooled over (protein, GO)
    instances, as in general_annotation_stats.py."""
    lens = [lengths[p] for p in ids if p in lengths]
    per_prot = [len(fan[p]) for p in ids]
    counter = Counter(g for p in ids for g in fan[p])
    ic_values = [ic_map[g] for g, c in counter.items() if g in ic_map for _ in range(c)]
    ns_inst = Counter()
    for g, c in counter.items():
        ns = ns_map.get(g)
        if ns:
            ns_inst[ns] += c
    ns_total = sum(ns_inst.values())
    return {
        f"N_{label}": len(ids),
        f"N_with_length_{label}": len(lens),
        f"Mean_length_{label}": _mean(lens),
        f"Median_length_{label}": _median(lens),
        f"Pct_short_lt100aa_{label}": 100.0 * sum(1 for x in lens if x < 100) / len(lens) if lens else float("nan"),
        f"GO_per_protein_mean_{label}": _mean(per_prot),
        f"GO_per_protein_median_{label}": _median(per_prot),
        f"N_unique_GO_{label}": len(counter),
        f"Total_GO_instances_{label}": sum(counter.values()),
        f"Mean_IC_{label}": _mean(ic_values),
        f"Median_IC_{label}": _median(ic_values),
        f"Pct_BP_{label}": 100.0 * ns_inst["BP"] / ns_total if ns_total else float("nan"),
        f"Pct_CC_{label}": 100.0 * ns_inst["CC"] / ns_total if ns_total else float("nan"),
        f"Pct_MF_{label}": 100.0 * ns_inst["MF"] / ns_total if ns_total else float("nan"),
    }


# ----------------------------------------------------------------- worker
_W = {}   # worker globals set by _init_worker (shared via fork, not pickled)


def _init_worker(ic_map: dict, ns_map: dict, proteome_glob: str, per_species_dir: Path, force: bool) -> None:
    _W.update(ic_map=ic_map, ns_map=ns_map, proteome_glob=proteome_glob,
              per_species_dir=per_species_dir, force=force)


def process_species(entry: dict) -> dict:
    """One species: classify proteins, count GOs, compute stats, write the
    per-species checkpoint JSON. Returns a short status dict for the log."""
    species = entry["species"]
    out_path = _W["per_species_dir"] / f"{species}.json"
    if not _W["force"] and out_path.exists() and out_path.stat().st_size > 0:
        return {"species": species, "status": "checkpoint"}
    try:
        fan_path = Path(entry["fantasia_file"])
        fan = parse_fantasia(fan_path)
        proteome, hom = parse_ahrd(Path(entry["homology_file"]))

        only_fan = [p for p in fan if p not in hom]
        both = [p for p in fan if p in hom]
        n_only_hom = sum(1 for p in hom if p not in fan)
        proteome_set = set(proteome)
        n_unannotated = sum(1 for p in proteome if p not in fan and p not in hom)
        n_fan_not_in_proteome = sum(1 for p in fan if p not in proteome_set)

        proteome_files = find_proteomes(fan_path, _W["proteome_glob"])
        lengths = {}
        for pf in proteome_files:
            parse_fasta_lengths(pf, lengths)

        counts = {
            "fantasia_only": Counter(g for p in only_fan for g in fan[p]),
            "fantasia_both": Counter(g for p in both for g in fan[p]),
            "homology_all": Counter(g for gos in hom.values() for g in gos),
        }

        n_prot = len(proteome)
        row = {
            "Species": species,
            "Group": entry["group"],
            "N_proteome": n_prot,
            "N_fantasia": len(fan),
            "N_homology_GO": len(hom),
            "N_only_fantasia": len(only_fan),
            "N_both": len(both),
            "N_only_homology": n_only_hom,
            "N_unannotated": n_unannotated,
            "Pct_only_fantasia_of_proteome": 100.0 * len(only_fan) / n_prot if n_prot else float("nan"),
            "Pct_only_fantasia_of_fantasia": 100.0 * len(only_fan) / len(fan) if fan else float("nan"),
            "Pct_both_of_proteome": 100.0 * len(both) / n_prot if n_prot else float("nan"),
            "Pct_fantasia_of_proteome": 100.0 * len(fan) / n_prot if n_prot else float("nan"),
            "Pct_homology_GO_of_proteome": 100.0 * len(hom) / n_prot if n_prot else float("nan"),
        }
        row.update(group_stats(only_fan, fan, lengths, _W["ic_map"], _W["ns_map"], "only_fantasia"))
        row.update(group_stats(both, fan, lengths, _W["ic_map"], _W["ns_map"], "both"))

        prot_lens = [lengths[p] for p in proteome if p in lengths]
        row["Mean_length_proteome"] = _mean(prot_lens)
        row["Median_length_proteome"] = _median(prot_lens)
        row["Length_MWU_p_only_vs_both"] = _length_pvalue(
            [lengths[p] for p in only_fan if p in lengths],
            [lengths[p] for p in both if p in lengths])

        # GO vocabulary exclusive to each group (within this species)
        go_only = set(counts["fantasia_only"])
        go_both = set(counts["fantasia_both"])
        row["N_GO_exclusive_only_fantasia"] = len(go_only - go_both)
        row["N_GO_exclusive_both"] = len(go_both - go_only)
        row["Pct_GO_exclusive_only_fantasia"] = 100.0 * len(go_only - go_both) / len(go_only) if go_only else float("nan")

        # FANTASIA vs homology agreement on the 'both' proteins
        hom_per_prot = [len(hom[p]) for p in both]
        jaccards = []
        for p in both:
            a, b = set(fan[p]), set(hom[p])
            jaccards.append(len(a & b) / len(a | b))
        row["Homology_GO_per_protein_mean_both"] = _mean(hom_per_prot)
        row["Jaccard_fantasia_homology_mean_both"] = _mean(jaccards)

        row["N_fantasia_ids_not_in_proteome"] = n_fan_not_in_proteome
        row["N_proteome_files"] = len(proteome_files)
        row["Proteome_files"] = ";".join(str(p) for p in proteome_files)

        with open(out_path, "w") as fh:
            json.dump({"stats": row, "counts": {k: dict(v) for k, v in counts.items()}}, fh)
        return {"species": species, "status": "ok", "n_proteome_files": len(proteome_files),
                "n_fan_not_in_proteome": n_fan_not_in_proteome}
    except Exception as exc:   # report, don't kill the pool
        return {"species": species, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ merging
def load_manifests(paths: list) -> list:
    """Rows with fantasia_status == homology_status == OK from the status
    TSVs; group = file stem (non_viridi / viridi). Logs the excluded rows."""
    entries, excluded = [], Counter()
    for path in paths:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        group = path.stem
        ok = (df["fantasia_status"] == "OK") & (df["homology_status"] == "OK")
        for _, r in df[~ok].iterrows():
            excluded[(group, r["fantasia_status"], r["homology_status"])] += 1
        for _, r in df[ok].iterrows():
            entries.append({"species": r["species"], "group": group,
                            "fantasia_file": r["fantasia_file"], "homology_file": r["homology_file"]})
        _log(f"  {path.name}: {int(ok.sum())} species OK/OK, {int((~ok).sum())} excluded")
    for (group, fs, hs), n in sorted(excluded.items()):
        _log(f"    excluded {group}: fantasia={fs} homology={hs} -> {n} species")
    dup = [s for s, c in Counter(e["species"] for e in entries).items() if c > 1]
    if dup:
        _log(f"  [WARN] {len(dup)} species names appear in more than one manifest; keeping the first occurrence")
        seen, uniq = set(), []
        for e in entries:
            if e["species"] not in seen:
                seen.add(e["species"])
                uniq.append(e)
        entries = uniq
    return entries


def build_wide_matrix(species_order: list, per_species_counts: dict) -> pd.DataFrame:
    """Species x GO integer matrix (columns sorted by GO id, rows in
    species_order), filled from {species: {GO: count}}."""
    go_ids = sorted({g for c in per_species_counts.values() for g in c})
    go_idx = {g: i for i, g in enumerate(go_ids)}
    mat = np.zeros((len(species_order), len(go_ids)), dtype=np.int32)
    for i, sp in enumerate(species_order):
        for g, c in per_species_counts.get(sp, {}).items():
            mat[i, go_idx[g]] = c
    df = pd.DataFrame(mat, index=pd.Index(species_order, name="Species"), columns=go_ids)
    return df


# ------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build dark-proteome (FANTASIA-only) and shared (FANTASIA+homology) "
                    "GO count matrices, a homology GO count matrix and per-species group "
                    "statistics from paired FANTASIA / AHRD annotation files.")
    ap.add_argument("--manifest", type=Path, nargs="+", required=True,
                    help="Status TSV(s) with columns species, fantasia_status, fantasia_file, "
                         "homology_status, homology_file (e.g. darkproteome/non_viridi.tsv "
                         "darkproteome/viridi.tsv). File stem is used as Group.")
    ap.add_argument("--output", type=Path, required=True, help="Output directory")
    ap.add_argument("--ic", type=Path, default=DEFAULT_IC_PATH,
                    help=f"GO IC table with namespace (default: {DEFAULT_IC_PATH})")
    ap.add_argument("--proteome_glob", default=DEFAULT_PROTEOME_GLOB,
                    help="Glob, relative to each species directory, locating the proteome "
                         f"FASTA used for protein lengths (default: {DEFAULT_PROTEOME_GLOB})")
    ap.add_argument("--threads", type=int, default=4, help="Species processed in parallel (default: 4)")
    ap.add_argument("--skip_matrices", action="store_true",
                    help="Skip Module 1 — do not write the three count matrices")
    ap.add_argument("--skip_stats", action="store_true",
                    help="Skip Module 2 — do not write the per-species statistics table")
    ap.add_argument("--force", action="store_true",
                    help="Rerun all steps from scratch even if intermediate outputs exist in workdir/")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--disable_co2_tracking", action="store_true",
                    help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()
    t_start = time.monotonic()

    args.manifest = [p.resolve() for p in args.manifest]
    args.ic = args.ic.resolve()
    _validate_inputs([("--manifest", p) for p in args.manifest] + [("--ic", args.ic)])

    run_dir = Path(args.output)
    results = run_dir / "results"
    workdir = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    per_species_dir = workdir / "per_species"
    for d in (results, workdir, logs_dir, per_species_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name

    global _LOG_FH
    log_path = _dated_log_path(logs_dir, "Run_DarkProteomeMatrices")
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  DarkProteomeMatrices {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    _banner(f"DarkProteomeMatrices {VERSION}")
    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif any(per_species_dir.iterdir()):
        _log("Existing workdir found — resuming from checkpoints (use --force to rerun all steps from scratch)")

    _log("Reading manifests")
    entries = load_manifests(args.manifest)
    _log(f"  {len(entries)} species to process")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        for p in args.manifest:
            _log(f"  Manifest    : {p}")
        _log(f"  IC table    : {args.ic}")
        _log(f"  Proteomes   : <species_dir>/{args.proteome_glob}")
        _log(f"  Output      : {run_dir}/")
        _log(f"  Threads     : {args.threads}")
        _log("  Steps that would run:")
        _log(f"    [0] Per-species classification + counts ({len(entries)} species)  →  workdir/per_species/*.json")
        if not args.skip_matrices:
            _log(f"    [1] Module 1 — count matrices  →  results/mod01_*_counts_{prefix}.tsv")
        if not args.skip_stats:
            _log(f"    [2] Module 2 — group statistics  →  results/mod02_dark_proteome_stats_{prefix}.tsv")
        _log(f"    [3] Module 3 — Group/Species table  →  results/mod03_taxons_{prefix}.tsv")
        _log("  Exiting (--dry_run).")
        _LOG_FH.close()
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="DarkProteomeMatrices", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped (conda install -c conda-forge codecarbon)")

    _log(f"Loading IC / namespace table: {args.ic}")
    ic_map, ns_map = load_ic_and_namespace(args.ic)
    _log(f"  {len(ic_map)} GO terms with IC, {len(ns_map)} with namespace")

    # ---- per-species pass (checkpointed per species) -----------------------
    _banner("Per-species classification and GO counting")
    status = Counter()
    errors, no_proteome, id_mismatch = [], [], []
    n_total = len(entries)
    with Pool(processes=max(1, args.threads), initializer=_init_worker,
              initargs=(ic_map, ns_map, args.proteome_glob, per_species_dir, args.force)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_species, entries, chunksize=1), 1):
            status[res["status"]] += 1
            if res["status"] == "error":
                errors.append(res)
                _log(f"  [{i}/{n_total}] ERROR {res['species']}: {res['error']}")
            elif res["status"] == "ok":
                if res["n_proteome_files"] == 0:
                    no_proteome.append(res["species"])
                if res["n_fan_not_in_proteome"] > 0:
                    id_mismatch.append((res["species"], res["n_fan_not_in_proteome"]))
            if i % 100 == 0 or i == n_total:
                _log(f"  [{i}/{n_total}] done: {status['ok']} processed, {status['checkpoint']} from checkpoint, "
                     f"{status['error']} errors")
    if no_proteome:
        _log(f"  [WARN] {len(no_proteome)} species without a proteome FASTA match (lengths NA): "
             f"{', '.join(no_proteome[:10])}{' ...' if len(no_proteome) > 10 else ''}")
    if id_mismatch:
        _log(f"  [WARN] {len(id_mismatch)} species with FANTASIA protein IDs absent from the AHRD table "
             f"(see N_fantasia_ids_not_in_proteome): "
             + ", ".join(f"{s} ({n})" for s, n in id_mismatch[:10]) + (" ..." if len(id_mismatch) > 10 else ""))

    # ---- collect ----------------------------------------------------------
    _log("Collecting per-species results")
    species_order, stats_rows = [], []
    counts = {"fantasia_only": {}, "fantasia_both": {}, "homology_all": {}}
    for e in entries:
        p = per_species_dir / f"{e['species']}.json"
        if not p.exists():
            continue
        with open(p) as fh:
            data = json.load(fh)
        species_order.append(e["species"])
        stats_rows.append(data["stats"])
        for k in counts:
            counts[k][e["species"]] = data["counts"][k]
    _log(f"  {len(species_order)} species with results")

    matrix_shapes = {}
    if not args.skip_matrices:
        _banner("Module 1 — count matrices")
        for key, fname in (("fantasia_only", "fantasia_only"), ("fantasia_both", "fantasia_both"),
                           ("homology_all", "homology_all")):
            out = results / f"mod01_{fname}_counts_{prefix}.tsv"
            df = build_wide_matrix(species_order, counts[key])
            df.to_csv(out, sep="\t")
            matrix_shapes[fname] = list(df.shape)
            _log(f"  {out.name}: {df.shape[0]} species x {df.shape[1]} GO terms, "
                 f"{int(df.to_numpy().sum())} GO instances")

    stats_df = pd.DataFrame(stats_rows)
    if not args.skip_stats:
        _banner("Module 2 — group statistics")
        out = results / f"mod02_dark_proteome_stats_{prefix}.tsv"
        stats_df.to_csv(out, sep="\t", index=False, float_format="%.4f")
        _log(f"  {out.name}: {len(stats_df)} species x {stats_df.shape[1]} columns")
        if len(stats_df):
            _log(f"  Pct_only_fantasia_of_proteome: median {stats_df['Pct_only_fantasia_of_proteome'].median():.2f}% "
                 f"(min {stats_df['Pct_only_fantasia_of_proteome'].min():.2f}%, "
                 f"max {stats_df['Pct_only_fantasia_of_proteome'].max():.2f}%)")
            _log(f"  Mean length only_fantasia vs both (median over species): "
                 f"{stats_df['Mean_length_only_fantasia'].median():.1f} vs {stats_df['Mean_length_both'].median():.1f} aa")

    _banner("Module 3 — Group/Species table")
    taxons = pd.DataFrame({"Group": [e["group"] for e in entries if e["species"] in set(species_order)],
                           "Species": species_order})
    taxons.to_csv(results / f"mod03_taxons_{prefix}.tsv", sep="\t", index=False)
    _log(f"  mod03_taxons_{prefix}.tsv: {len(taxons)} species")

    # ---- summary ----------------------------------------------------------
    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_manifests": [str(p) for p in args.manifest],
        "input_ic": str(args.ic),
        "n_species_in_manifests_ok": len(entries),
        "n_species_processed": len(species_order),
        "n_species_errors": len(errors),
        "errors": errors,
        "n_species_without_proteome": len(no_proteome),
        "matrix_shapes": matrix_shapes,
        "parameters": {"proteome_glob": args.proteome_glob, "threads": args.threads,
                       "homology_annotated_definition": "AHRD Gene-Ontology-Term column non-empty",
                       "skip_matrices": args.skip_matrices, "skip_stats": args.skip_stats, "force": args.force},
        "resource_usage": {"wall_clock_s": round(elapsed_s, 1), "peak_mem_mb": round(peak_mem_mb, 1),
                           "emissions_kg_CO2eq": emissions_kg},
    }
    with open(results / f"{prefix}.run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    _banner("Done")
    _log(f"  {len(species_order)} species, {len(errors)} errors, {elapsed_s/60:.1f} min, peak {peak_mem_mb:.0f} MB")
    _log(f"  Results: {results}/")
    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
