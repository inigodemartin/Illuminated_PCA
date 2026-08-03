#!/usr/bin/env python3
"""
Builds a full NCBI ranked lineage (superkingdom/kingdom/phylum/class/order/
family/genus) per species, for every species in data/species_taxid.tsv that
has a resolved taxid -- the foundation for generalizing the Metazoa-phylum
accordion in general_pca_template.html to any group / any taxonomic rank
(see build_ncbi_taxid_lookup.py, which resolves the taxid this script starts
from).

Source: NCBI's new_taxdump (https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/
new_taxdump/new_taxdump.tar.gz, ~150 MB compressed), specifically
rankedlineage.dmp -- one line per NCBI taxon id giving the name of its
ancestor at each of 7 fixed ranks directly (blank where NCBI's tree doesn't
assign that rank for a given lineage -- kingdom is commonly blank for algae/
protists, for instance), so no need to walk nodes.dmp by hand. Downloaded
once and cached locally (checkpointed, not committed to the repo -- ~1.3 GB
uncompressed between the two files extracted); only the small per-species
output TSV is bundled.

Species with no resolved taxid (see build_ncbi_taxid_lookup.py -- currently
all 436 Asgard MAG codes plus a handful of others) get a row with blank
lineage columns, same one-row-per-species shape as species_taxid.tsv, so
this file stays a straight 1:1 join against merged_taxons_belen.tsv. They're
meant to be excluded from taxonomy-driven views (nothing to show), while
still rendering normally in the plain PCA -- same "falls back to itself,
no per-species special-casing needed downstream" shape as
metazoa_subgroup in general_pca_abundance.py.
"""

VERSION = "v0.1.0"

import argparse
import getpass
import os
import platform
import resource
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"
RANKED_COLUMNS = ["superkingdom", "kingdom", "phylum", "class", "order", "family", "genus"]

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


# ------------------------------------------------------------------ taxdump
def download_taxdump(archive_path: Path, force: bool) -> None:
    if _checkpoint(archive_path, "new_taxdump.tar.gz", force):
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"  downloading {NCBI_TAXDUMP_URL} (~150 MB)...")
    t0 = time.monotonic()
    with requests.get(NCBI_TAXDUMP_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp_path = archive_path.with_suffix(".tar.gz.part")
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp_path.rename(archive_path)
    _log(f"  downloaded {archive_path.stat().st_size / 1e6:.0f} MB in {time.monotonic()-t0:.0f}s")


def extract_taxdump_files(archive_path: Path, dest_dir: Path, force: bool) -> tuple:
    members = ["rankedlineage.dmp", "merged.dmp"]
    paths = tuple(dest_dir / m for m in members)
    to_extract = [m for m, p in zip(members, paths) if not _checkpoint(p, m, force)]
    if to_extract:
        _log(f"  extracting {', '.join(to_extract)} from {archive_path.name}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_dir, members=[tar.getmember(m) for m in to_extract], filter="data")
        for p in paths:
            if p.exists():
                _log(f"    {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
    return paths


def load_merged_taxids(merged_path: Path) -> dict:
    """old (retired/merged-away) taxid -> the current taxid it was merged into."""
    out = {}
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t|\t")
            if len(fields) >= 2:
                out[int(fields[0])] = int(fields[1].rstrip("\t|"))
    return out


def load_wanted_lineages(rankedlineage_path: Path, wanted_taxids: set) -> dict:
    """
    taxid -> {tax_name, superkingdom, kingdom, phylum, class, order, family,
    genus}, for just the taxids in wanted_taxids. Streams the ~1.3 GB dmp
    file line by line rather than loading it whole -- only ~3900 of its
    ~2.6M lines are ever kept.

    rankedlineage.dmp columns (NCBI's fixed schema, "\t|\t"-separated):
    tax_id, tax_name, species, genus, family, order, class, phylum, kingdom,
    superkingdom. "species" is the ancestor name at species rank -- blank
    when tax_id itself already IS the species-rank node (as opposed to a
    strain/isolate below it), so not used here; genus upward is what feeds
    RANKED_COLUMNS.
    """
    out = {}
    n_found = 0
    with open(rankedlineage_path, encoding="utf-8") as fh:
        for line in fh:
            tab = line.index("\t")
            taxid_str = line[:tab]
            if not taxid_str.isdigit():
                continue
            taxid = int(taxid_str)
            if taxid not in wanted_taxids:
                continue
            fields = [f.strip() for f in line.rstrip("\n").split("\t|\t")]
            # fields: [tax_id, tax_name, species, genus, family, order, class, phylum, kingdom, superkingdom|]
            tax_name, _species, genus, family, order, class_, phylum, kingdom, superkingdom = fields[1:10]
            out[taxid] = {
                "tax_name": tax_name,
                "superkingdom": superkingdom.rstrip("|").strip(),
                "kingdom": kingdom,
                "phylum": phylum,
                "class": class_,
                "order": order,
                "family": family,
                "genus": genus,
            }
            n_found += 1
            if n_found == len(wanted_taxids):
                break
    return out


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-taxid", type=Path, default=Path(__file__).parent.parent / "data" / "species_taxid.tsv",
                     help="species -> taxid lookup (default: bundled data/species_taxid.tsv, "
                          "see build_ncbi_taxid_lookup.py)")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "species_lineage.tsv",
                     help="Output TSV: Species, Group, TaxID, TaxName, superkingdom..genus "
                          "(default: bundled data/species_lineage.tsv)")
    ap.add_argument("--taxdump-cache-dir", type=Path,
                     default=Path(__file__).parent.parent / "data" / ".ncbi_taxdump_cache",
                     help="Where to download/extract new_taxdump.tar.gz (default: bundled data/"
                          ".ncbi_taxdump_cache, gitignored -- ~1.3 GB, not part of the repo)")
    ap.add_argument("--force", action="store_true",
                     help="Re-download/re-extract the taxdump and re-run from scratch")
    ap.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    args.species_taxid = args.species_taxid.resolve()
    args.output = args.output.resolve()
    args.taxdump_cache_dir = args.taxdump_cache_dir.resolve()

    if not args.species_taxid.exists():
        print(f"ERROR: --species-taxid not found: {args.species_taxid}", file=sys.stderr)
        sys.exit(1)

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    global _LOG_FH
    log_path = logs_dir / "Run_BuildSpeciesLineage.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  BuildSpeciesLineage {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    roster = pd.read_csv(args.species_taxid, sep="\t")
    n_resolved = int(roster["TaxID"].notna().sum())

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  species_taxid.tsv    : {args.species_taxid} ({len(roster)} species, {n_resolved} with a taxid)")
        _log(f"  Taxdump cache dir    : {args.taxdump_cache_dir}")
        _log(f"  Output               : {args.output}")
        _log("  Steps that would run:")
        _log("    [1] Download new_taxdump.tar.gz (~150 MB)  → " + str(args.taxdump_cache_dir))
        _log("    [2] Extract rankedlineage.dmp               → " + str(args.taxdump_cache_dir))
        _log(f"    [3] Look up lineage for {n_resolved} taxids  → in-memory")
        _log("    [4] Write merged lineage TSV                → " + str(args.output))
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    if args.force:
        _log("--force set: taxdump will be re-downloaded/re-extracted and all species re-looked-up")

    t_start = time.monotonic()

    _banner("Paso 1 — Descargar new_taxdump")
    archive_path = args.taxdump_cache_dir / "new_taxdump.tar.gz"
    download_taxdump(archive_path, args.force)

    _banner("Paso 2 — Extraer rankedlineage.dmp y merged.dmp")
    rankedlineage_path, merged_path = extract_taxdump_files(archive_path, args.taxdump_cache_dir, args.force)

    _banner("Paso 3 — Buscar el linaje de cada taxid")
    wanted_taxids = set(int(t) for t in roster.loc[roster["TaxID"].notna(), "TaxID"])
    _log(f"  buscando {len(wanted_taxids)} taxids distintos en {rankedlineage_path.name}...")
    t0 = time.monotonic()
    lineages = load_wanted_lineages(rankedlineage_path, wanted_taxids)
    _log(f"  {len(lineages)}/{len(wanted_taxids)} taxids encontrados en el taxdump ({time.monotonic()-t0:.0f}s)")

    missing = wanted_taxids - set(lineages)
    merged_to = {}
    if missing:
        # A taxid resolved earlier (build_ncbi_taxid_lookup.py) can go
        # missing here simply because NCBI retired/merged it into another
        # taxid since -- merged.dmp maps the old id to its replacement, so
        # a second lookup usually recovers most of these for free.
        merged_map = load_merged_taxids(merged_path)
        merged_to = {t: merged_map[t] for t in missing if t in merged_map}
        if merged_to:
            recovered = load_wanted_lineages(rankedlineage_path, set(merged_to.values()))
            for old_taxid, new_taxid in merged_to.items():
                if new_taxid in recovered:
                    lineages[old_taxid] = recovered[new_taxid]
            n_recovered = sum(1 for t in merged_to if t in lineages)
            _log(f"  {n_recovered}/{len(merged_to)} de los que faltaban se recuperaron vía merged.dmp "
                 f"(taxid retirado → taxid actual)")
        still_missing = wanted_taxids - set(lineages)
        if still_missing:
            _log(f"  [WARN] {len(still_missing)} taxids siguen sin aparecer en el taxdump actual "
                 f"(ni directo ni vía merged.dmp): {sorted(still_missing)[:10]}"
                 + (" ..." if len(still_missing) > 10 else ""))

    _banner("Paso 4 — Escribiendo tabla de linaje")
    rows = []
    for _, row in roster.iterrows():
        taxid = row["TaxID"]
        lineage = lineages.get(int(taxid)) if pd.notna(taxid) else None
        out_row = {
            "Species": row["Species"],
            "Group": row["Group"],
            "TaxID": int(taxid) if pd.notna(taxid) else None,
            "TaxName": lineage["tax_name"] if lineage else None,
        }
        for rank in RANKED_COLUMNS:
            out_row[rank] = (lineage.get(rank) or None) if lineage else None
        rows.append(out_row)

    df = pd.DataFrame(rows, columns=["Species", "Group", "TaxID", "TaxName"] + RANKED_COLUMNS)
    df["TaxID"] = df["TaxID"].astype("Int64")
    df.to_csv(args.output, sep="\t", index=False)
    _log(f"  {len(df)} especies escritas → {args.output}")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    _banner("Listo")
    n_with_phylum = int(df["phylum"].notna().sum())
    _log(f"  Especies totales    : {len(df)}")
    _log(f"  Con linaje (phylum) : {n_with_phylum} ({n_with_phylum / len(df) * 100:.1f}%)")
    for rank in RANKED_COLUMNS:
        _log(f"    con {rank:13s}: {int(df[rank].notna().sum())}")
    _log(f"  Tiempo total        : {elapsed_s:.1f}s")
    _log(f"  Pico de memoria     : {peak_mem_mb:.1f} MB")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
