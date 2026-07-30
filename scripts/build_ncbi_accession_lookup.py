#!/usr/bin/env python3
"""
Builds a species -> NCBI genome accession lookup so the PCA/UMAP/PHATE HTML
pages can click straight through to a species' NCBI Datasets genome page
(https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/) instead of a
plain-text NCBI Genome search that often finds nothing (see general_pca_
template.html's openNcbiGenome).

Three sources, each species run through exactly one:

  - Fungi (non_viridiplantae): fungi_structure.txt is a `tree`-style listing
    of the FANTASIA_project run directory. Each species' 00_GenomeSource/
    has exactly one child directory named after the assembly accession
    (e.g. "GCF_010015735.1") -- that name IS the accession, verbatim.

  - Viridiplantae: viridiplantae_structure.txt, same tree shape, but
    00_GenomeSource/ is messier: sometimes the datasets-CLI layout (bare
    "GCA_xxx.x" dir alongside assembly_data_report.jsonl/dataset_catalog.
    json, same as Fungi), sometimes a manually-named "NCBI_GCA_xxx_x[_tag]"
    dir (dot replaced by underscore, optional suffix like "_RefSeq"), and
    sometimes a non-NCBI source entirely (CNCB/CNGB/HWBASE/Figshare/lab-
    specific dumps) that has no NCBI accession at all -- left unresolved.
    A handful of species have two candidate accessions (an old GCA_ plus a
    newer GCF_ RefSeq re-annotation); see resolve_viridiplantae_candidates.

  - Metazoa: these species were run by someone else and never had a local
    genome download, so there's no folder to read an accession from at all
    -- data/metadata_metazoa.txt only has an NCBI taxon id (ID_NCBI) per
    species. Resolved via the public NCBI Datasets REST API's per-taxon
    reference-genome lookup (no API key needed, just rate-limited to
    ~3 req/s). Cached to disk (one JSON blob keyed by taxon id) so re-runs
    without --force are instant.

Asgard (internal ASG### codes, no scientific name anywhere) and Protists
(informal/environmental-sample names, no taxon id in the metadata) have no
reliable NCBI identifier available at all -- left out of the lookup table on
purpose. Species missing from the output TSV simply keep the old
text-search fallback in the HTML.
"""

VERSION = "v0.1.0"

import argparse
import getpass
import json
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from general_pca_common import DEFAULT_FUNGI_STRUCTURE_PATH, DEFAULT_VIRIDIPLANTAE_STRUCTURE_PATH  # noqa: E402

NCBI_API_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"
NCBI_REQUEST_INTERVAL_S = 0.34  # public rate limit is ~3 req/s without an API key

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


# --------------------------------------------------------- tree-file parsing
# fungi_structure.txt / viridiplantae_structure.txt are `tree`-style
# listings: each depth level is a fixed 4-char block, either "│   " (still
# has siblings below) or "    " (no more siblings), followed by "├── " or
# "└── " for the actual entry. The blank-fill characters are non-breaking
# spaces (U+00A0), not regular spaces -- normalize before matching.
_LINE_RE = re.compile(r"^((?:[│ ]   )*)(├── |└── )(.*)$")


def _depth_and_name(line: str):
    line = line.replace(" ", " ")
    m = _LINE_RE.match(line)
    if not m:
        return None
    prefix, _branch, name = m.groups()
    depth = len(prefix) // 4 + 1
    return depth, name


def collect_genome_source_children(tree_path: Path) -> dict:
    """species name -> list of direct children of its 00_GenomeSource/ dir."""
    species = None
    depth2 = None
    children = {}
    with open(tree_path, encoding="utf-8") as fh:
        for line in fh:
            parsed = _depth_and_name(line.rstrip("\n"))
            if parsed is None:
                continue
            depth, name = parsed
            if depth == 1:
                species = name
                depth2 = None
            elif depth == 2:
                depth2 = name
            elif depth == 3 and depth2 == "00_GenomeSource":
                children.setdefault(species, []).append(name)
    return children


_BARE_ACCESSION_RE = re.compile(r"^(GCA|GCF)_\d+\.\d+$")
_NCBI_TAGGED_ACCESSION_RE = re.compile(r"^NCBI_(GCA|GCF)_(\d+)_(\d+)")


def _candidate_accessions(entries: list) -> list:
    """[(raw_entry_name, normalized_accession), ...] for entries that look like an NCBI accession."""
    out = []
    for entry in entries:
        m = _BARE_ACCESSION_RE.match(entry)
        if m:
            out.append((entry, entry))
            continue
        m = _NCBI_TAGGED_ACCESSION_RE.match(entry)
        if m:
            out.append((entry, f"{m.group(1)}_{m.group(2)}.{m.group(3)}"))
    return out


def resolve_viridiplantae_candidates(entries: list) -> str:
    """
    Pick one accession when a species' 00_GenomeSource/ has more than one
    NCBI-shaped candidate (old GCA_ download left alongside a newer GCF_
    RefSeq re-annotation, or two versions of the same accession). Prefer,
    in order: a folder name tagged "RefSeq" > a GCF_ (RefSeq) accession over
    GCA_ (GenBank) > the lexicographically-highest accession (highest
    version/newest accession number) as a last tiebreaker.
    """
    candidates = _candidate_accessions(entries)
    uniq = sorted(set(acc for _raw, acc in candidates))
    if len(uniq) <= 1:
        return uniq[0] if uniq else None

    refseq_tagged = sorted({acc for raw, acc in candidates if "RefSeq" in raw})
    if len(refseq_tagged) == 1:
        return refseq_tagged[0]
    pool = refseq_tagged if refseq_tagged else uniq

    gcf_only = sorted(acc for acc in pool if acc.startswith("GCF_"))
    pool = gcf_only if gcf_only else pool

    return max(pool)


def parse_fungi_accessions(tree_path: Path) -> dict:
    children_by_species = collect_genome_source_children(tree_path)
    result = {}
    n_missing = 0
    for species, entries in children_by_species.items():
        uniq = sorted(set(acc for _raw, acc in _candidate_accessions(entries)))
        if not uniq:
            n_missing += 1
            continue
        if len(uniq) > 1:
            _log(f"  [WARN] fungi: {species} has {len(uniq)} accession candidates in 00_GenomeSource "
                 f"({uniq}), using the first")
        result[species] = uniq[0]
    _log(f"  fungi ({tree_path.name}): {len(result)}/{len(children_by_species)} species resolved to an accession"
         + (f", {n_missing} had no NCBI-shaped entry under 00_GenomeSource" if n_missing else ""))
    return result


def parse_viridiplantae_accessions(tree_path: Path) -> dict:
    children_by_species = collect_genome_source_children(tree_path)
    result = {}
    n_missing = 0
    n_ambiguous = 0
    for species, entries in children_by_species.items():
        candidates = _candidate_accessions(entries)
        if not candidates:
            n_missing += 1
            continue
        if len({acc for _raw, acc in candidates}) > 1:
            n_ambiguous += 1
        result[species] = resolve_viridiplantae_candidates(entries)
    _log(f"  viridiplantae ({tree_path.name}): {len(result)}/{len(children_by_species)} species resolved to an "
         f"accession ({n_ambiguous} had >1 candidate, resolved by preference rule; "
         f"{n_missing} had no NCBI-shaped entry -- non-NCBI source, e.g. CNCB/CNGB/Figshare)")
    return result


# ----------------------------------------------------------------- Metazoa
def load_metazoa_taxon_ids(metadata_path: Path) -> dict:
    """species name (SCIENTIFIC_NAME with spaces -> underscores) -> NCBI taxon id.

    Species-name join must match merge_fantasia_species.py's discover_metazoa
    exactly (sci_name.strip().replace(" ", "_")) so the lookup keys line up
    with the species names already used throughout the PCA/UMAP/PHATE pages.
    """
    meta = pd.read_csv(metadata_path, sep="\t")
    out = {}
    for _, row in meta.iterrows():
        species = str(row["SCIENTIFIC_NAME"]).strip().replace(" ", "_")
        out[species] = int(row["ID_NCBI"])
    return out


def fetch_reference_accession(taxon_id: int, session: requests.Session) -> str:
    """
    Reference (or, failing that, representative) genome accession for an
    NCBI taxon id, or None if the taxon has no assembly at all -- both
    perfectly normal for a taxon id with no sequenced genome.
    """
    url = f"{NCBI_API_BASE}/genome/taxon/{taxon_id}/dataset_report"
    resp = session.get(url, params={"filters.reference_only": "true"}, timeout=30)
    resp.raise_for_status()
    reports = resp.json().get("reports", [])
    if not reports:
        return None
    return reports[0].get("accession")


def resolve_metazoa_accessions(taxon_ids: dict, cache_path: Path, force: bool) -> dict:
    """
    species -> accession, resolved one taxon id at a time against the NCBI
    Datasets API. Cached to `cache_path` as {taxon_id: accession_or_None} so
    re-running this script doesn't re-issue ~1000 requests (each request is
    rate-limited to keep under NCBI's public 3 req/s cap).
    """
    cache = {}
    if cache_path.exists():
        cache = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
        _log(f"  loaded {len(cache)} cached taxon lookups from {cache_path.name}")

    unique_taxa = sorted(set(taxon_ids.values()))
    pending = [t for t in unique_taxa if force or t not in cache]
    _log(f"  {len(unique_taxa)} distinct Metazoa taxon ids, {len(pending)} need an API lookup "
         f"({len(unique_taxa) - len(pending)} already cached)")

    if pending:
        session = requests.Session()
        t0 = time.monotonic()
        for i, taxon_id in enumerate(pending, 1):
            try:
                cache[taxon_id] = fetch_reference_accession(taxon_id, session)
            except requests.RequestException as exc:
                _log(f"  [WARN] taxon {taxon_id}: NCBI API request failed ({exc}), leaving unresolved")
                cache[taxon_id] = None
            if i % 50 == 0 or i == len(pending):
                cache_path.write_text(json.dumps(cache, indent=2))
                _log(f"    ... {i}/{len(pending)} taxa queried ({time.monotonic()-t0:.0f}s), checkpointed")
            time.sleep(NCBI_REQUEST_INTERVAL_S)
        cache_path.write_text(json.dumps(cache, indent=2))

    n_resolved = sum(1 for v in cache.values() if v)
    _log(f"  Metazoa: {n_resolved}/{len(unique_taxa)} taxon ids have a reference genome on NCBI")

    return {species: cache.get(taxon_id) for species, taxon_id in taxon_ids.items() if cache.get(taxon_id)}


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fungi-structure", type=Path, default=DEFAULT_FUNGI_STRUCTURE_PATH,
                     help="tree listing of the non_viridiplantae FANTASIA run dir (default: bundled fungi_structure.txt)")
    ap.add_argument("--viridiplantae-structure", type=Path, default=DEFAULT_VIRIDIPLANTAE_STRUCTURE_PATH,
                     help="tree listing of the viridiplantae FANTASIA run dir (default: bundled viridiplantae_structure.txt)")
    ap.add_argument("--metazoa-metadata", type=Path, default=Path(__file__).parent.parent / "data" / "metadata_metazoa.txt",
                     help="metadata_metazoa.txt (columns ID, SCIENTIFIC_NAME, ID_NCBI, Phylum)")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "species_ncbi_accession.tsv",
                     help="Output TSV: Species, NCBI_Accession, Source (default: bundled data/species_ncbi_accession.tsv)")
    ap.add_argument("--metazoa-cache", type=Path, default=Path(__file__).parent.parent / "data" / "ncbi_taxon_accession_cache.json",
                     help="Taxon-id -> accession API response cache (default: bundled data/ncbi_taxon_accession_cache.json)")
    ap.add_argument("--skip_fungi", action="store_true", help="Skip Module 1 — Fungi/non_viridiplantae tree parsing")
    ap.add_argument("--skip_viridiplantae", action="store_true", help="Skip Module 2 — Viridiplantae tree parsing")
    ap.add_argument("--skip_metazoa", action="store_true", help="Skip Module 3 — Metazoa NCBI API resolution")
    ap.add_argument("--force", action="store_true", help="Re-run all steps from scratch, ignoring the Metazoa API cache")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    args.fungi_structure = args.fungi_structure.resolve()
    args.viridiplantae_structure = args.viridiplantae_structure.resolve()
    args.metazoa_metadata = args.metazoa_metadata.resolve()
    args.output = args.output.resolve()
    args.metazoa_cache = args.metazoa_cache.resolve()

    missing = []
    if not args.skip_fungi and not args.fungi_structure.exists():
        missing.append(("--fungi-structure", args.fungi_structure))
    if not args.skip_viridiplantae and not args.viridiplantae_structure.exists():
        missing.append(("--viridiplantae-structure", args.viridiplantae_structure))
    if not args.skip_metazoa and not args.metazoa_metadata.exists():
        missing.append(("--metazoa-metadata", args.metazoa_metadata))
    if missing:
        for flag, p in missing:
            print(f"ERROR: {flag} not found: {p}", file=sys.stderr)
        sys.exit(1)

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    global _LOG_FH
    log_path = logs_dir / "Run_BuildNcbiAccessionLookup.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  BuildNcbiAccessionLookup {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    if args.force:
        _log("--force set: Metazoa API cache will be ignored and re-queried in full")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Fungi structure         : {'skip' if args.skip_fungi else args.fungi_structure}")
        _log(f"  Viridiplantae structure : {'skip' if args.skip_viridiplantae else args.viridiplantae_structure}")
        _log(f"  Metazoa metadata        : {'skip' if args.skip_metazoa else args.metazoa_metadata}")
        _log(f"  Output                  : {args.output}")
        _log("  Steps that would run:")
        if not args.skip_fungi:
            _log("    [1] Parse Fungi tree           → in-memory species->accession")
        if not args.skip_viridiplantae:
            _log("    [2] Parse Viridiplantae tree   → in-memory species->accession")
        if not args.skip_metazoa:
            _log("    [3] Resolve Metazoa via NCBI API → checkpointed to data/ncbi_taxon_accession_cache.json")
        _log("    [4] Write merged lookup TSV    → " + str(args.output))
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    t_start = time.monotonic()
    rows = []

    if not args.skip_fungi:
        _banner("Módulo 1 — Fungi (non_viridiplantae)")
        for species, accession in parse_fungi_accessions(args.fungi_structure).items():
            rows.append({"Species": species, "NCBI_Accession": accession, "Source": "fungi_structure"})

    if not args.skip_viridiplantae:
        _banner("Módulo 2 — Viridiplantae")
        for species, accession in parse_viridiplantae_accessions(args.viridiplantae_structure).items():
            rows.append({"Species": species, "NCBI_Accession": accession, "Source": "viridiplantae_structure"})

    if not args.skip_metazoa:
        _banner("Módulo 3 — Metazoa (vía NCBI Datasets API)")
        taxon_ids = load_metazoa_taxon_ids(args.metazoa_metadata)
        for species, accession in resolve_metazoa_accessions(taxon_ids, args.metazoa_cache, args.force).items():
            rows.append({"Species": species, "NCBI_Accession": accession, "Source": "metazoa_api"})

    _banner("Escribiendo tabla de lookup")
    ran_sources = {
        "fungi_structure": not args.skip_fungi,
        "viridiplantae_structure": not args.skip_viridiplantae,
        "metazoa_api": not args.skip_metazoa,
    }
    # Modules can be run independently (e.g. --skip_metazoa while re-parsing
    # the tree files, or vice versa) -- carry over rows from any source that
    # didn't run this time rather than dropping them from the output file.
    if args.output.exists():
        existing = pd.read_csv(args.output, sep="\t")
        carried_over = existing[~existing["Source"].map(lambda s: ran_sources.get(s, False))]
        if len(carried_over):
            _log(f"  carrying over {len(carried_over)} rows from sources not re-run this time "
                 f"({sorted(carried_over['Source'].unique())})")
        rows = carried_over.to_dict("records") + rows

    df = pd.DataFrame(rows, columns=["Species", "NCBI_Accession", "Source"]).sort_values("Species")
    n_dup = int(df["Species"].duplicated().sum())
    if n_dup:
        _log(f"  [WARN] {n_dup} especies duplicadas entre fuentes — se conserva la primera aparición")
        df = df.drop_duplicates(subset="Species", keep="first")
    df.to_csv(args.output, sep="\t", index=False)
    _log(f"  {len(df)} especies con accession NCBI → {args.output}")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    _banner("Listo")
    _log(f"  Especies resueltas : {len(df)}")
    _log(f"  Tiempo total       : {elapsed_s:.1f}s")
    _log(f"  Pico de memoria    : {peak_mem_mb:.1f} MB")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
