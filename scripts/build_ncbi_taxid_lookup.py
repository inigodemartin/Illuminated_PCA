#!/usr/bin/env python3
"""
Builds a species -> NCBI taxonomy id (taxid) lookup for every species in
merged_taxons_belen.tsv, as the foundation for taxonomy-driven features in
the PCA/UMAP/PHATE pages (e.g. generalizing the Metazoa-phylum accordion in
general_pca_template.html to any group / any taxonomic rank, or a "color by
rank" view) -- those need a full NCBI lineage per species, which in turn
needs a taxid per species to look the lineage up from.

Three sources, tried in order, each species resolved by exactly one:

  1. Any species with a resolved genome accession in
     data/species_ncbi_accession.tsv (built by build_ncbi_accession_lookup.py
     -- mostly Fungi/Viridiplantae, plus whichever Metazoa/Protists/
     Rhodophyta happened to get one too): the accession's dataset report
     includes organism.tax_id directly, exact and unambiguous. Queried in
     batches of ~100 accessions per request (NCBI Datasets API v2 accepts a
     comma-separated accession list in the URL, up to a URL-length limit --
     see BATCH_SIZE) instead of one request per species, so this whole step
     is under 30 HTTP requests for ~2700 species.

  2. Metazoa: data/metadata_metazoa.txt already has an NCBI taxon id
     (ID_NCBI) per species -- no API call needed at all. Only used as a
     fallback for Metazoa species step 1 couldn't resolve (no accession),
     not tried first, because that file has at least one confirmed wrong
     ID_NCBI (Synaphobranchus_kaupii -> 1862, which is actually the
     bacterial genus Dermatophilus, not the fish -- caught downstream via
     its "Bacillati" kingdom in species_lineage.tsv). An accession-verified
     taxid is trusted over this file's ID_NCBI whenever both are available.

  3. Everything else: NCBI Taxonomy name search (E-utils esearch, db=
     taxonomy), one request per species (rate-limited same as step 1). Tries
     the species' full display name first (catches strain-level "Candidatus"-
     style taxa where the strain suffix is actually part of the accepted
     name), then falls back to just the leading two tokens (genus + species)
     if that finds nothing (catches names carrying an isolate/strain suffix
     NCBI taxonomy never registered on its own).

Asgard (informal ASG### MAG codes -- no scientific name anywhere, see
data/metadata_asgard.txt which only has a phylum-level Group label, e.g.
"Thor") has no reliable per-species NCBI identifier available at all, same
conclusion build_ncbi_accession_lookup.py already reached for its accession
lookup -- skipped without even attempting a name search (which would just
burn ~436 requests to confirm zero hits), left unresolved on purpose.
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
import requests

NCBI_DATASETS_API_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_REQUEST_INTERVAL_S = 0.34  # public rate limit is ~3 req/s without an API key
ACCESSION_BATCH_SIZE = 100  # comma-joined accessions per Datasets API request; ~150 is the URL-length ceiling

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


# --------------------------------------------------------------- Step 1: Metazoa
def load_metazoa_taxon_ids(metadata_path: Path) -> dict:
    """species name (SCIENTIFIC_NAME with spaces -> underscores) -> NCBI taxon id.

    Same species-name join as build_ncbi_accession_lookup.py's
    load_metazoa_taxon_ids, so the keys line up with merged_taxons_belen.tsv.
    """
    meta = pd.read_csv(metadata_path, sep="\t")
    out = {}
    for _, row in meta.iterrows():
        species = str(row["SCIENTIFIC_NAME"]).strip().replace(" ", "_")
        out[species] = int(row["ID_NCBI"])
    return out


# --------------------------------------------------------- Step 2: accessions
def load_accessions(accession_path: Path) -> dict:
    """species -> NCBI genome accession, from build_ncbi_accession_lookup.py's output."""
    df = pd.read_csv(accession_path, sep="\t")
    return dict(zip(df["Species"], df["NCBI_Accession"]))


def fetch_taxids_for_accessions(accessions: list, session: requests.Session) -> dict:
    """accession -> taxid, for one batch (already <= ACCESSION_BATCH_SIZE)."""
    url = f"{NCBI_DATASETS_API_BASE}/genome/accession/{','.join(accessions)}/dataset_report"
    resp = session.get(url, params={"page_size": 1000}, timeout=60)
    resp.raise_for_status()
    out = {}
    for report in resp.json().get("reports", []):
        acc = report.get("accession")
        tax_id = report.get("organism", {}).get("tax_id")
        if acc and tax_id:
            out[acc] = int(tax_id)
    return out


def resolve_accession_taxids(accessions: dict, cache_path: Path, force: bool) -> dict:
    """
    species -> taxid, resolved via NCBI Datasets API in batches of
    ACCESSION_BATCH_SIZE accessions per request. Cached to cache_path as
    {accession: taxid} so re-running without --force is instant.
    """
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        _log(f"  loaded {len(cache)} cached accession->taxid lookups from {cache_path.name}")

    unique_accessions = sorted(set(accessions.values()))
    pending = [a for a in unique_accessions if force or a not in cache]
    _log(f"  {len(unique_accessions)} distinct accessions, {len(pending)} need an API lookup "
         f"({len(unique_accessions) - len(pending)} already cached)")

    if pending:
        session = requests.Session()
        t0 = time.monotonic()
        batches = [pending[i:i + ACCESSION_BATCH_SIZE] for i in range(0, len(pending), ACCESSION_BATCH_SIZE)]
        for i, batch in enumerate(batches, 1):
            try:
                resolved = fetch_taxids_for_accessions(batch, session)
                for acc in batch:
                    cache[acc] = resolved.get(acc)
            except requests.RequestException as exc:
                _log(f"  [WARN] accession batch {i}/{len(batches)}: NCBI API request failed ({exc}), "
                     f"leaving {len(batch)} accessions unresolved")
                for acc in batch:
                    cache.setdefault(acc, None)
            cache_path.write_text(json.dumps(cache, indent=2))
            _log(f"    ... batch {i}/{len(batches)} ({len(batch)} accessions, "
                 f"{time.monotonic()-t0:.0f}s), checkpointed")
            time.sleep(NCBI_REQUEST_INTERVAL_S)

    n_resolved = sum(1 for v in cache.values() if v)
    _log(f"  accessions: {n_resolved}/{len(unique_accessions)} resolved to a taxid")

    return {species: cache.get(acc) for species, acc in accessions.items() if cache.get(acc)}


# ---------------------------------------------------------- Step 3: name search
def search_taxid_by_name(display_name: str, session: requests.Session) -> tuple:
    """
    Returns (taxid_or_None, method) where method is "full_name",
    "binomial_fallback", or None (nothing found either way).

    Tries the full display name first (unrestricted "All Names" field --
    the only way to catch strain-level "Candidatus ..." taxa whose accepted
    scientific name legitimately includes what looks like a strain suffix),
    then falls back to just the leading two tokens (genus + species) with
    the stricter [Scientific Name] field, which avoids substring false-
    positives that an unrestricted search on just two common words could
    otherwise pick up.
    """
    def esearch(term: str):
        resp = session.get(f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                            params={"db": "taxonomy", "term": term, "retmode": "json", "retmax": 5},
                            timeout=30)
        resp.raise_for_status()
        result = resp.json().get("esearchresult", {})
        ids = result.get("idlist", [])
        return ids

    full_term = display_name.replace("_", " ").strip()
    ids = esearch(full_term)
    if ids:
        if len(ids) > 1:
            _log(f"  [WARN] '{full_term}' matched {len(ids)} taxonomy ids, using the first ({ids[0]})")
        return int(ids[0]), "full_name"

    tokens = full_term.split()
    if len(tokens) > 2:
        binomial = " ".join(tokens[:2])
        time.sleep(NCBI_REQUEST_INTERVAL_S)
        ids = esearch(f"{binomial}[Scientific Name]")
        if ids:
            if len(ids) > 1:
                _log(f"  [WARN] '{binomial}' matched {len(ids)} taxonomy ids, using the first ({ids[0]})")
            return int(ids[0]), "binomial_fallback"

    return None, None


def resolve_name_search_taxids(species_list: list, cache_path: Path, force: bool) -> dict:
    """
    species -> (taxid, method), resolved one at a time against NCBI Taxonomy
    (no batch endpoint for name search). Cached to cache_path as
    {species: [taxid_or_None, method_or_None]} so re-running without
    --force is instant.
    """
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        _log(f"  loaded {len(cache)} cached name-search lookups from {cache_path.name}")

    pending = [s for s in species_list if force or s not in cache]
    _log(f"  {len(species_list)} species need a name search, {len(pending)} need an API lookup "
         f"({len(species_list) - len(pending)} already cached)")

    if pending:
        session = requests.Session()
        t0 = time.monotonic()
        for i, species in enumerate(pending, 1):
            try:
                taxid, method = search_taxid_by_name(species, session)
                cache[species] = [taxid, method]
            except requests.RequestException as exc:
                _log(f"  [WARN] '{species}': NCBI API request failed ({exc}), leaving unresolved")
                cache[species] = [None, None]
            if i % 50 == 0 or i == len(pending):
                cache_path.write_text(json.dumps(cache, indent=2))
                _log(f"    ... {i}/{len(pending)} species queried ({time.monotonic()-t0:.0f}s), checkpointed")
            time.sleep(NCBI_REQUEST_INTERVAL_S)
        cache_path.write_text(json.dumps(cache, indent=2))

    n_resolved = sum(1 for v in cache.values() if v[0])
    _log(f"  name search: {n_resolved}/{len(species_list)} species resolved to a taxid")

    return {s: tuple(cache[s]) for s in species_list if s in cache and cache[s][0]}


# ---------------------------------------------------------------- CLI / main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", type=Path, default=Path(__file__).parent.parent / "merged_taxons_belen.tsv",
                     help="Species roster with Group column (default: bundled merged_taxons_belen.tsv)")
    ap.add_argument("--metazoa-metadata", type=Path, default=Path(__file__).parent.parent / "data" / "metadata_metazoa.txt",
                     help="metadata_metazoa.txt (columns ID, SCIENTIFIC_NAME, ID_NCBI, Phylum)")
    ap.add_argument("--accessions", type=Path, default=Path(__file__).parent.parent / "data" / "species_ncbi_accession.tsv",
                     help="species -> NCBI accession lookup (default: bundled data/species_ncbi_accession.tsv, "
                          "see build_ncbi_accession_lookup.py)")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "species_taxid.tsv",
                     help="Output TSV: Species, Group, TaxID, Source (default: bundled data/species_taxid.tsv)")
    ap.add_argument("--accession-cache", type=Path,
                     default=Path(__file__).parent.parent / "data" / "ncbi_accession_taxid_cache.json",
                     help="accession -> taxid API response cache")
    ap.add_argument("--name-search-cache", type=Path,
                     default=Path(__file__).parent.parent / "data" / "ncbi_name_search_taxid_cache.json",
                     help="species -> taxid name-search API response cache")
    ap.add_argument("--skip_metazoa", action="store_true", help="Skip Step 1 — Metazoa metadata lookup")
    ap.add_argument("--skip_accessions", action="store_true", help="Skip Step 2 — accession-based NCBI API resolution")
    ap.add_argument("--skip_name_search", action="store_true", help="Skip Step 3 — taxonomy name-search fallback")
    ap.add_argument("--force", action="store_true", help="Re-run all steps from scratch, ignoring API caches")
    ap.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print the steps that would run, then exit without executing anything")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()

    args.taxonomy = args.taxonomy.resolve()
    args.metazoa_metadata = args.metazoa_metadata.resolve()
    args.accessions = args.accessions.resolve()
    args.output = args.output.resolve()
    args.accession_cache = args.accession_cache.resolve()
    args.name_search_cache = args.name_search_cache.resolve()

    missing = []
    if not args.taxonomy.exists():
        missing.append(("--taxonomy", args.taxonomy))
    if not args.skip_metazoa and not args.metazoa_metadata.exists():
        missing.append(("--metazoa-metadata", args.metazoa_metadata))
    if not args.skip_accessions and not args.accessions.exists():
        missing.append(("--accessions", args.accessions))
    if missing:
        for flag, p in missing:
            print(f"ERROR: {flag} not found: {p}", file=sys.stderr)
        sys.exit(1)

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    global _LOG_FH
    log_path = logs_dir / "Run_BuildNcbiTaxidLookup.log"
    _LOG_FH = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  BuildNcbiTaxidLookup {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    if args.force:
        _log("--force set: API caches will be ignored and re-queried in full")

    roster = pd.read_csv(args.taxonomy, sep="\t")
    all_species = dict(zip(roster["Species"], roster["Group"]))

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Taxonomy roster   : {args.taxonomy} ({len(all_species)} species)")
        _log(f"  Metazoa metadata  : {'skip' if args.skip_metazoa else args.metazoa_metadata}")
        _log(f"  Accessions        : {'skip' if args.skip_accessions else args.accessions}")
        _log(f"  Output            : {args.output}")
        _log("  Steps that would run:")
        if not args.skip_accessions:
            _log("    [1] Accession-based NCBI API    → checkpointed to " + args.accession_cache.name)
        if not args.skip_metazoa:
            _log("    [2] Metazoa metadata lookup     → in-memory species->taxid (no API calls)")
        if not args.skip_name_search:
            _log("    [3] Taxonomy name-search fallback → checkpointed to " + args.name_search_cache.name)
        _log("    [4] Write merged lookup TSV     → " + str(args.output))
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    t_start = time.monotonic()
    resolved = {}  # species -> (taxid, source)

    if not args.skip_accessions:
        _banner("Paso 1 — Especies con accession NCBI (vía NCBI Datasets API, en lotes)")
        accessions_all = load_accessions(args.accessions)
        pending_accessions = {sp: acc for sp, acc in accessions_all.items()
                               if sp in all_species and sp not in resolved}
        _log(f"  {len(pending_accessions)} especies con accession pendientes de resolver "
             f"(de {len(accessions_all)} con accession en total)")
        for species, taxid in resolve_accession_taxids(pending_accessions, args.accession_cache, args.force).items():
            resolved[species] = (taxid, "accession_api")
        _log(f"  {len(resolved)} especies resueltas vía accession")

    if not args.skip_metazoa:
        _banner("Paso 2 — Metazoa sin accession (metadata_metazoa.txt, sin llamadas a la API)")
        taxon_ids = load_metazoa_taxon_ids(args.metazoa_metadata)
        n_before = len(resolved)
        for species, taxid in taxon_ids.items():
            if species in all_species and species not in resolved:
                resolved[species] = (taxid, "metazoa_metadata")
        _log(f"  {len(resolved) - n_before} especies adicionales resueltas desde metadata_metazoa.txt "
             f"({n_before} ya resueltas vía accession en el paso 1, sin sobreescribir)")

    if not args.skip_name_search:
        _banner("Paso 3 — Búsqueda por nombre en NCBI Taxonomy (resto de especies)")
        pending_names = sorted(sp for sp, group in all_species.items()
                                if sp not in resolved and group != "Asgard")
        n_asgard = sum(1 for sp, g in all_species.items() if g == "Asgard" and sp not in resolved)
        if n_asgard:
            _log(f"  {n_asgard} especies de Asgard excluidas a propósito (códigos MAG informales "
                 f"ASG###, sin nombre científico -- ver metadata_asgard.txt, que solo da un filo, "
                 f"no un taxid por especie). Quedan sin resolver.")
        for species, (taxid, method) in resolve_name_search_taxids(pending_names, args.name_search_cache, args.force).items():
            resolved[species] = (taxid, f"name_search_{method}")

    _banner("Escribiendo tabla de lookup")
    rows = []
    for species, group in all_species.items():
        taxid, source = resolved.get(species, (None, "unresolved"))
        rows.append({"Species": species, "Group": group, "TaxID": taxid, "Source": source})
    df = pd.DataFrame(rows, columns=["Species", "Group", "TaxID", "Source"]).sort_values("Species")
    # Int64 (nullable) instead of the default float64 -- a plain int dtype
    # can't represent the missing TaxIDs for unresolved species, and without
    # this every resolved id would otherwise round-trip through CSV as
    # "1450172.0" instead of "1450172".
    df["TaxID"] = df["TaxID"].astype("Int64")
    df.to_csv(args.output, sep="\t", index=False)
    _log(f"  {len(df)} especies escritas → {args.output}")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)

    _banner("Listo")
    n_resolved = int(df["TaxID"].notna().sum())
    _log(f"  Especies totales   : {len(df)}")
    _log(f"  Con taxid          : {n_resolved} ({n_resolved / len(df) * 100:.1f}%)")
    for source, count in df["Source"].value_counts().items():
        _log(f"    {source:25s}: {count}")
    _log(f"  Tiempo total       : {elapsed_s:.1f}s")
    _log(f"  Pico de memoria    : {peak_mem_mb:.1f} MB")

    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
