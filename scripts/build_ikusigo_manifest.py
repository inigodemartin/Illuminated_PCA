#!/usr/bin/env python3
"""
Prepares IkusiGO inputs (see /home/inigo/claude/IkusiGO/) from this project's
existing FANTASIA annotation runs: collects each species' GO annotation file
into one flat folder, renamed "{Species}_FANTASIA_topGO.tsv", and writes the
manifest TSV (Species, TaxID, AnnotationFile) IkusiGO expects.

Covers Viridiplantae and non_viridiplantae (Fungi + everything else FANTASIA
was run on directly) only -- Asgard/Metazoa/Protists annotations were run by
someone else and handed over already organized, so this script doesn't touch
them (see the user's own framing: "no les tenemos que hacer nada").

Scans --viridiplantae-root / --non-viridiplantae-root directly on disk --
these are the real filesystem directories containing each species' folder
(e.g. /data/.../FANTASIA_project/, NOT bundled with this repo). No auxiliary
tree-listing file needed; the two *_structure.txt files in this repo were
only ever a reference for exploring the layout while writing this script,
never a runtime input.

Two modules:

  1. Resolve  -- walks each species' <root>/<species>/04_FunctionalAnnotation/
                 for FANTASIA_2025* folder(s) and, inside each, a file
                 matching "{prefix}_GOs_merged.tsv". A species can have more
                 than one FANTASIA_2025* folder with a valid annotation
                 inside (e.g. Arabidopsis_thaliana has 16 -- different
                 gene-model sources: Araport11, TAIR10, BRAKER, ...) --
                 prompts interactively which one to use in that case
                 (--non_interactive skips those instead of prompting).

  2. Copy + manifest -- for every species Module 1 resolved, copies its
                 chosen annotation file into results/annotations/, looks up
                 its NCBI taxid in --species-taxid (default: bundled
                 data/species_taxid.tsv), and writes
                 results/manifest_{prefix}.tsv. Species with no resolved
                 TaxID are copied but excluded from the manifest (logged in
                 results/skipped_species_{prefix}.tsv) since IkusiGO
                 requires one.
"""

VERSION = "v0.1.0"

import argparse
import getpass
import os
import platform
import re
import resource
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_SPECIES_TAXID = Path(__file__).parent.parent / "data" / "species_taxid.tsv"

FANTASIA_DIR_RE = re.compile(r"^FANTASIA_2025(_.+)?$")
GOS_MERGED_RE = re.compile(r"^(\w+)_GOs_merged\.tsv$")

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


# --------------------------------------------------------- Module 1: resolve
def collect_fantasia_dirs(root: Path) -> dict:
    """
    species -> {FANTASIA_2025* dir name: [filenames directly inside]}, read
    straight off disk under <root>/<species>/04_FunctionalAnnotation/ --
    other FANTASIA-ish folders (plain "FANTASIA", "FANTASIAv4", ...) are
    deliberately not matched, only the 2025 reruns. A species folder with no
    04_FunctionalAnnotation/ subdirectory at all is silently skipped (not
    every entry under --root need be a species with annotation, e.g. stray
    logs/scratch dirs).
    """
    result = {}
    for species_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        func_annot_dir = species_dir / "04_FunctionalAnnotation"
        if not func_annot_dir.is_dir():
            continue
        dirs = {}
        for candidate in sorted(p for p in func_annot_dir.iterdir() if p.is_dir()):
            if FANTASIA_DIR_RE.match(candidate.name):
                dirs[candidate.name] = [p.name for p in candidate.iterdir()]
        if dirs:
            result[species_dir.name] = dirs
    return result


def valid_candidates(dirs: dict) -> dict:
    """FANTASIA_2025* dir name -> its "{prefix}_GOs_merged.tsv" filename, for dirs that have one."""
    out = {}
    for dirname, children in dirs.items():
        for child in children:
            if GOS_MERGED_RE.match(child):
                out[dirname] = child
                break
    return out


def prompt_choice(species: str, candidates: dict):
    """Interactively ask which FANTASIA_2025* folder to use. Returns (dirname, filename) or None to skip."""
    items = list(candidates.items())
    print(f"\n[ambiguous] {species} has {len(items)} valid FANTASIA_2025* annotation folders:")
    for i, (dirname, filename) in enumerate(items, 1):
        print(f"  {i}. {dirname}  ({filename})")
    print("  0. skip this species")
    while True:
        choice = input(f"  Choose 1-{len(items)} (0 = skip): ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1]
        print("  invalid choice, try again")


def resolve_group(root: Path, group_label: str, non_interactive: bool) -> list:
    dirs_by_species = collect_fantasia_dirs(root)
    rows = []
    n_auto = n_multi = n_zero = 0
    for species in sorted(dirs_by_species):
        candidates = valid_candidates(dirs_by_species[species])
        if not candidates:
            n_zero += 1
            rows.append({"Species": species, "Group": group_label, "FantasiaDir": None,
                         "AnnotationFilename": None, "Status": "no_valid_annotation"})
            continue
        if len(candidates) == 1:
            n_auto += 1
            (dirname, filename), = candidates.items()
        else:
            n_multi += 1
            if non_interactive:
                rows.append({"Species": species, "Group": group_label, "FantasiaDir": None,
                             "AnnotationFilename": None, "Status": "ambiguous_skipped_non_interactive"})
                continue
            choice = prompt_choice(species, candidates)
            if choice is None:
                rows.append({"Species": species, "Group": group_label, "FantasiaDir": None,
                             "AnnotationFilename": None, "Status": "skipped_by_user"})
                continue
            dirname, filename = choice
        rows.append({"Species": species, "Group": group_label, "FantasiaDir": dirname,
                     "AnnotationFilename": filename, "Status": "resolved"})
    _log(f"  {group_label} ({root}): {len(dirs_by_species)} species with a FANTASIA_2025* folder — "
         f"{n_auto} auto-resolved, {n_multi} ambiguous, {n_zero} with no valid annotation")
    return rows


def run_module1(resolved_path: Path, viridiplantae_root: Path, non_viridiplantae_root: Path,
                 skip_viridiplantae: bool, skip_non_viridiplantae: bool, non_interactive: bool, force: bool) -> pd.DataFrame:
    if _checkpoint(resolved_path, "resolved annotation paths", force):
        return pd.read_csv(resolved_path, sep="\t")

    rows = []
    ran_groups = set()
    if not skip_viridiplantae:
        rows += resolve_group(viridiplantae_root, "viridiplantae", non_interactive)
        ran_groups.add("viridiplantae")
    if not skip_non_viridiplantae:
        rows += resolve_group(non_viridiplantae_root, "non_viridiplantae", non_interactive)
        ran_groups.add("non_viridiplantae")

    df = pd.DataFrame(rows, columns=["Species", "Group", "FantasiaDir", "AnnotationFilename", "Status"])
    # A group skipped this run (e.g. --skip_viridiplantae while re-resolving
    # non_viridiplantae) keeps its previously-resolved rows instead of
    # dropping them -- avoids re-prompting for the same ambiguous species
    # every time you re-run for just one group.
    if resolved_path.exists():
        existing = pd.read_csv(resolved_path, sep="\t")
        carried = existing[~existing["Group"].isin(ran_groups)]
        if len(carried):
            _log(f"  carrying over {len(carried)} rows from group(s) not re-resolved this run: "
                 f"{sorted(carried['Group'].unique())}")
        df = pd.concat([carried, df], ignore_index=True)

    df.to_csv(resolved_path, sep="\t", index=False)
    _log(f"  wrote {resolved_path.name} ({len(df)} rows)")
    return df


# --------------------------------------------------------- Module 2: copy
def run_module2(resolved_df: pd.DataFrame, roots: dict, species_taxid_path: Path,
                 results_dir: Path, prefix: str, force: bool) -> None:
    manifest_path = results_dir / f"manifest_{prefix}.tsv"
    if _checkpoint(manifest_path, "manifest", force):
        return

    taxid_df = pd.read_csv(species_taxid_path, sep="\t")
    taxid_map = dict(zip(taxid_df["Species"], taxid_df["TaxID"]))

    annotations_dir = results_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    skipped_rows = []
    n_copied = 0
    for _, row in resolved_df.iterrows():
        species = row["Species"]
        if row["Status"] != "resolved":
            skipped_rows.append({"Species": species, "Reason": row["Status"]})
            continue

        root = roots.get(row["Group"])
        if root is None:
            skipped_rows.append({"Species": species, "Reason": f"no root given for group '{row['Group']}'"})
            continue

        src = root / species / "04_FunctionalAnnotation" / row["FantasiaDir"] / row["AnnotationFilename"]
        if not src.exists():
            skipped_rows.append({"Species": species, "Reason": f"source file not found: {src}"})
            continue

        dest_filename = f"{species}_FANTASIA_topGO.tsv"
        shutil.copy2(src, annotations_dir / dest_filename)
        n_copied += 1

        taxid = taxid_map.get(species)
        if taxid is None or pd.isna(taxid):
            skipped_rows.append({"Species": species,
                                 "Reason": "copied, but no TaxID in species_taxid.tsv — add manually to the manifest"})
            continue
        manifest_rows.append({"Species": species, "TaxID": int(taxid), "AnnotationFile": dest_filename})

    manifest_df = pd.DataFrame(manifest_rows, columns=["Species", "TaxID", "AnnotationFile"])
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    skipped_path = results_dir / f"skipped_species_{prefix}.tsv"
    pd.DataFrame(skipped_rows, columns=["Species", "Reason"]).to_csv(skipped_path, sep="\t", index=False)

    _log(f"  copied {n_copied} annotation files → {annotations_dir}/")
    _log(f"  wrote {manifest_path.name} ({len(manifest_df)} species)")
    _log(f"  wrote {skipped_path.name} ({len(skipped_rows)} species skipped/excluded — see Reason column)")


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-taxid", type=Path, default=DEFAULT_SPECIES_TAXID,
                     help="Species -> TaxID lookup TSV (default: bundled data/species_taxid.tsv)")
    ap.add_argument("--viridiplantae-root", type=Path, default=None,
                     help="Real filesystem directory containing each Viridiplantae species' folder "
                          "(required unless --skip_viridiplantae)")
    ap.add_argument("--non-viridiplantae-root", type=Path, default=None,
                     help="Real filesystem directory containing each non_viridiplantae species' folder "
                          "(required unless --skip_non_viridiplantae)")
    ap.add_argument("--output", type=Path, required=True, help="Run output directory")
    ap.add_argument("--skip_viridiplantae", action="store_true", help="Skip the Viridiplantae group entirely")
    ap.add_argument("--skip_non_viridiplantae", action="store_true", help="Skip the non_viridiplantae group entirely")
    ap.add_argument("--skip_module1", action="store_true",
                     help="Skip Module 1 — reuse the existing workdir/mod01_resolved_*.tsv (no re-parsing/re-prompting)")
    ap.add_argument("--skip_module2", action="store_true",
                     help="Skip Module 2 — resolve and report only, don't copy files or write the manifest")
    ap.add_argument("--non_interactive", action="store_true",
                     help="Skip ambiguous species (>1 valid FANTASIA_2025* folder) instead of prompting")
    ap.add_argument("--force", action="store_true",
                     help="Rerun all steps from scratch even if intermediate outputs exist in workdir/")
    ap.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    args.species_taxid = args.species_taxid.resolve()
    args.output = args.output.resolve()
    if args.viridiplantae_root:
        args.viridiplantae_root = args.viridiplantae_root.resolve()
    if args.non_viridiplantae_root:
        args.non_viridiplantae_root = args.non_viridiplantae_root.resolve()

    missing = []
    if not args.skip_module2 and not args.species_taxid.exists():
        missing.append(("--species-taxid", args.species_taxid))
    if args.viridiplantae_root is not None and not args.viridiplantae_root.is_dir():
        missing.append(("--viridiplantae-root", args.viridiplantae_root))
    if args.non_viridiplantae_root is not None and not args.non_viridiplantae_root.is_dir():
        missing.append(("--non-viridiplantae-root", args.non_viridiplantae_root))
    if missing:
        for flag, p in missing:
            print(f"ERROR: {flag} not found: {p}", file=sys.stderr)
        sys.exit(1)

    results = args.output / "results"
    workdir = args.output / "workdir"
    logs_dir = args.output / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = args.output.name

    global _LOG_FH
    log_path = logs_dir / "Run_BuildIkusiGOManifest.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  BuildIkusiGOManifest {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Viridiplantae root          : {'skip' if args.skip_viridiplantae else args.viridiplantae_root}")
        _log(f"  non_viridiplantae root      : {'skip' if args.skip_non_viridiplantae else args.non_viridiplantae_root}")
        _log(f"  Output                      : {args.output}/")
        _log("  Steps that would run:")
        if not args.skip_module1:
            _log("    [1] Resolve annotation paths (may prompt for ambiguous species) → workdir/mod01_resolved_*.tsv")
        if not args.skip_module2:
            _log("    [2] Copy annotation files + write manifest → results/annotations/, results/manifest_*.tsv")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    root_needed = not (args.skip_module1 and args.skip_module2)
    if root_needed and not args.skip_viridiplantae and args.viridiplantae_root is None:
        print("ERROR: --viridiplantae-root is required (or pass --skip_viridiplantae)", file=sys.stderr)
        sys.exit(1)
    if root_needed and not args.skip_non_viridiplantae and args.non_viridiplantae_root is None:
        print("ERROR: --non-viridiplantae-root is required (or pass --skip_non_viridiplantae)", file=sys.stderr)
        sys.exit(1)

    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif workdir.exists() and any(workdir.iterdir()):
        _log("Existing workdir found — resuming from checkpoints (use --force to rerun all steps from scratch)")

    t_start = time.monotonic()

    _banner("Module 1 — Resolve annotation paths")
    resolved_path = workdir / f"mod01_resolved_{prefix}.tsv"
    if args.skip_module1:
        if not resolved_path.exists():
            print(f"ERROR: --skip_module1 set but {resolved_path} doesn't exist yet — run Module 1 at least once first",
                  file=sys.stderr)
            sys.exit(1)
        _log("  --skip_module1 set: loading existing resolved-paths table from workdir")
        resolved_df = pd.read_csv(resolved_path, sep="\t")
    else:
        resolved_df = run_module1(
            resolved_path, args.viridiplantae_root, args.non_viridiplantae_root,
            args.skip_viridiplantae, args.skip_non_viridiplantae, args.non_interactive, args.force,
        )

    if not args.skip_module2:
        _banner("Module 2 — Copy annotation files + write manifest")
        roots = {"viridiplantae": args.viridiplantae_root, "non_viridiplantae": args.non_viridiplantae_root}
        run_module2(resolved_df, roots, args.species_taxid, results, prefix, args.force)

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    _banner("Done")
    _log(f"  Wall-clock time : {elapsed_s:.1f}s")
    _log(f"  Peak memory     : {peak_mem_mb:.1f} MB")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
