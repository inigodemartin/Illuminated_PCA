#!/usr/bin/env python3
"""
Shared helpers for the standalone-HTML general PCA scripts (presence_absence_pca.py,
general_pca_abundance.py): both render the same template and need the same GO
search/illumination payload, just fit on different PCA inputs.
"""

from pathlib import Path
import base64
import colorsys
import gzip

import numpy as np
import pandas as pd

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_pca_template.html"
DEFAULT_IC_PATH = Path(__file__).parent.parent / "data" / "All_GOs_ic.tsv"
DEFAULT_FUNGI_STRUCTURE_PATH = Path(__file__).parent.parent / "fungi_structure.txt"
DEFAULT_VIRIDIPLANTAE_STRUCTURE_PATH = Path(__file__).parent.parent / "viridiplantae_structure.txt"
DEFAULT_SPECIES_ACCESSION_PATH = Path(__file__).parent.parent / "data" / "species_ncbi_accession.tsv"
DEFAULT_SPECIES_LINEAGE_PATH = Path(__file__).parent.parent / "data" / "species_lineage.tsv"
DEFAULT_METAZOA_TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "metadata_metazoa.txt"
DATA_MARKER = "__GENERAL_PCA_DATA__"
TITLE_MARKER = "__GENERAL_PCA_TITLE__"

# NCBI ranks species_lineage.tsv carries (see build_species_lineage.py) that
# are useful to subdivide a taxonomic group by in the browser -- superkingdom
# excluded, since every species in this dataset except Asgard (no taxid at
# all, see build_ncbi_taxid_lookup.py) is Eukaryota, so it never has more
# than one distinct value within any of our groups.
LINEAGE_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Ranks offered by the browser's "Subdivide by" control -- kingdom excluded
# since assign_kingdom_grouping() already uses it as the legend's top level,
# where subdividing a kingdom bucket by kingdom would always be a no-op.
# NOT currently wired into general_pca_abundance.py/presence_absence_pca.py
# (see METAZOA_SUBDIVIDE_RANKS/load_metazoa_phylum_lineage below, which
# reinstated the original curated-Group/Metazoa-only-phylum behavior) --
# kept so the full any-group/any-rank accordion can be recovered later just
# by switching those two scripts back to assign_kingdom_grouping/
# load_species_lineage/SUBDIVIDE_RANKS.
SUBDIVIDE_RANKS = [rank for rank in LINEAGE_RANKS if rank != "kingdom"]

# The single rank fed to the same "subdivide by rank" browser mechanism
# above when restricting it back to its original scope: only the Metazoa
# group, only by phylum (see load_metazoa_phylum_lineage). A single-entry
# rank list makes the "Subdivide by" <select> pointless, so the template
# hides that control whenever LINEAGE_RANKS.length <= 1 -- the control's
# code isn't removed, just dormant until SUBDIVIDE_RANKS (or another
# multi-rank list) is fed again.
METAZOA_SUBDIVIDE_RANKS = ["phylum"]


def load_species_ncbi_accessions(path):
    """
    species name -> NCBI accession (e.g. "GCF_010015735.1"), for the species
    that scripts/build_ncbi_accession_lookup.py could resolve one for.
    Species missing from this table (Asgard, Protists, and any Fungi/
    Viridiplantae/Metazoa species without a resolvable accession) just get
    {} back for that key, which the HTML falls back to a text search for.
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["Species"], df["NCBI_Accession"]))


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def load_species_lineage(path=DEFAULT_SPECIES_LINEAGE_PATH):
    """
    species -> {rank: value, ...} for each rank in LINEAGE_RANKS the species
    has a value for (blank/missing ranks -- e.g. kingdom is commonly blank
    for algae/protists in NCBI's tree, and every rank is blank for a species
    with no resolved taxid at all, currently all of Asgard -- are simply
    omitted from that species' dict rather than stored as ""). Missing file,
    or a species absent from it entirely, both just mean {} for that
    species: lets the browser's "subdivide by rank" control treat "no
    lineage data for this species" uniformly as "not part of any
    subdivision", without a separate has-data check downstream.
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    out = {}
    for _, row in df.iterrows():
        lineage = {rank: row[rank] for rank in LINEAGE_RANKS if pd.notna(row.get(rank))}
        if lineage:
            out[row["Species"]] = lineage
    return out


def load_metazoa_phylum_lineage(path=DEFAULT_METAZOA_TAXONOMY_PATH):
    """
    species -> {"phylum": <value>}, for every Metazoa species in
    data/metadata_metazoa.txt (columns ID, SCIENTIFIC_NAME, ID_NCBI, Phylum)
    that has a Phylum value -- the original, Metazoa-only counterpart of
    load_species_lineage() above, restored because the curated 12-group
    taxonomy's Fungi/Metazoa/Protists/... top level (merged_taxons_belen.tsv)
    is the one actually wanted at the legend's top level, with only Metazoa
    getting a phylum-level accordion, not a full kingdom-anchored, any-rank
    one. SCIENTIFIC_NAME uses "Genus species" (space-separated); species
    elsewhere in this pipeline are "Genus_species" (underscore-joined), so
    the join key is normalized here. Shape-compatible with
    build_lineage_label_colors() and the browser's generic subdivisionsFor()/
    isExpandable(), which is why this reuses that same {rank: value} dict
    shape instead of a flat species->phylum mapping.
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    out = {}
    for _, row in df.iterrows():
        species = str(row["SCIENTIFIC_NAME"]).replace(" ", "_")
        phylum = row.get("Phylum")
        if pd.notna(phylum):
            out[species] = {"phylum": phylum}
    return out


def assign_kingdom_grouping(species_records, species_lineage):
    """
    Repoints each record's group/base_group from the hand-curated taxonomy
    Group column to its real NCBI kingdom, so the legend's top-level rows
    are the real kingdom and the existing per-rank accordion (see
    SUBDIVIDE_RANKS) subdivides those kingdom buckets exactly as it already
    subdivided the curated groups. A species with no kingdom value (no
    resolved taxid at all -- currently all of Asgard -- or a kingdom-less
    lineage, e.g. several algae/protist groups in NCBI's tree) keeps its
    curated Group name as a fallback top-level bucket instead of being
    merged into a single meaningless "no data" catch-all that would mix
    unrelated lineages together. The original curated value survives in
    "original_group" for anything downstream that still wants it (e.g.
    go_group_diff.py's --group-a/--group-b).
    """
    for rec in species_records:
        lineage = species_lineage.get(rec["name"], {})
        for rank in LINEAGE_RANKS:
            if rank in lineage:
                rec[rank] = lineage[rank]
        rec["original_group"] = rec["group"]
        rec["base_group"] = lineage.get("kingdom") or rec["group"]
        rec["group"] = rec["base_group"]


def build_lineage_label_colors(species_lineage):
    """
    Stable, distinct colors for every lineage label that appears at any rank
    for any species (a phylum name, a class name, ...) -- offset into a hue
    range starting opposite build_global_color_map's (which starts at hue
    0), so a lineage label's color reads as visually distinct from the
    existing taxon-group palette instead of coincidentally landing near one
    of them. One shared palette across all ranks (not one per rank) since
    only one rank's worth of labels is ever shown at a time in the legend.
    """
    labels = sorted({value for lineage in species_lineage.values() for value in lineage.values()})
    n = len(labels)
    colors = {}
    for i, label in enumerate(labels):
        hue = ((i / n) + 0.5) % 1.0 if n else 0.0
        saturation = 0.65 + (i % 3) * 0.1
        colors[label] = rgb_to_hex(colorsys.hsv_to_rgb(hue, saturation, 0.9))
    return colors


def load_go_ic_and_descriptions(ic_file):
    """
    Single-pass read of the headerless IC TSV, returning both dicts.
    Format: go_id, category, col3, col4, ic, description, trailing-tab.
    """
    ic, desc = {}, {}
    with open(ic_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[0] in ic:
                continue
            try:
                ic[parts[0]] = float(parts[4])
            except ValueError:
                pass
            desc[parts[0]] = parts[5]
    return ic, desc


def load_go_descriptions(ic_file):
    """GO -> description. Wrapper around load_go_ic_and_descriptions."""
    _, desc = load_go_ic_and_descriptions(ic_file)
    return desc


def load_go_ic(ic_file):
    """GO id -> IC value (float). Wrapper around load_go_ic_and_descriptions."""
    ic, _ = load_go_ic_and_descriptions(ic_file)
    return ic


def top_loadings_by_pc(loadings, go_desc, n):
    """
    For each PC, the n GO terms with the most positive loadings and the n
    with the most negative loadings -- returned as {"positive": [...], "negative": [...]}.
    """
    result = {}
    for pc in loadings.columns:
        col = loadings[pc]
        result[pc] = {
            "positive": [
                {"go_id": go_id, "description": go_desc.get(go_id, "unknown"), "loading": float(v)}
                for go_id, v in col.nlargest(n).items()
            ],
            "negative": [
                {"go_id": go_id, "description": go_desc.get(go_id, "unknown"), "loading": float(v)}
                for go_id, v in col.nsmallest(n).items()
            ],
        }
    return result


def write_top_loadings_tsv(top_loadings, output_path):
    rows = []
    for pc, sections in top_loadings.items():
        for direction, entries in sections.items():
            for rank, entry in enumerate(entries, start=1):
                rows.append({
                    "PC": pc,
                    "Direction": direction,
                    "Rank": rank,
                    "GO_id": entry["go_id"],
                    "Description": entry["description"],
                    "Loading": entry["loading"],
                })
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def write_full_loadings_tsv(loadings, go_desc, output_path):
    """
    Full per-GO-term loading matrix -- one row per GO id retained in the
    PCA, one column per principal component fit, unlike write_top_loadings_tsv
    which only keeps the n most positive/negative per PC.
    """
    df = loadings.copy()
    df.insert(0, "Description", [go_desc.get(go_id, "unknown") for go_id in df.index])
    df.insert(0, "GO_id", df.index)
    df.to_csv(output_path, sep="\t", index=False)


def compute_species_contributions(normalized_df, loadings, n=10):
    """
    For each species, the n GO terms whose normalized-abundance × loading
    product contributes most positively and most negatively to each PC score.
    This tells you *why* a species sits where it does: which GO terms pull
    it along each axis and in which direction.

    Uses numpy broadcasting to compute the full (n_species × n_go) contribution
    matrix per PC in one shot rather than looping per species.
    """
    norm_arr = normalized_df.to_numpy()          # (n_species, n_go)
    go_ids = normalized_df.columns.tolist()
    species_list = normalized_df.index.tolist()

    result = {}
    for pc in loadings.columns:
        load_arr = loadings[pc].to_numpy()        # (n_go,)
        contrib = norm_arr * load_arr             # (n_species, n_go)

        n_safe = min(n, len(go_ids))
        for i, name in enumerate(species_list):
            row = contrib[i]
            top_pos = np.argpartition(row, -n_safe)[-n_safe:]
            top_pos = top_pos[np.argsort(row[top_pos])[::-1]]
            top_neg = np.argpartition(row, n_safe)[:n_safe]
            top_neg = top_neg[np.argsort(row[top_neg])]

            sp = result.setdefault(name, {})
            sp[pc] = {
                "positive": [{"go_id": go_ids[j], "contribution": round(float(row[j]), 5)} for j in top_pos],
                "negative": [{"go_id": go_ids[j], "contribution": round(float(row[j]), 5)} for j in top_neg],
                "total_positive": round(float(np.sum(row[row > 0])), 5),
                "total_negative": round(float(np.sum(row[row < 0])), 5),
            }
    return result


def build_go_search_payload(raw_full, species, go_desc):
    """
    Lets the browser look up, for any GO id in the *full* matrix (not just
    the rare-term-filtered PCA input -- searching/illuminating a term
    shouldn't be limited by the PCA's own numerical-stability filter), its
    raw count for every currently-plotted species. Mirrors what
    illuminate_PCA.run_illuminated_PCA does for a single queried GO term,
    just generalized to "any GO term, picked live in the browser" instead
    of one baked in at generation time.

    There's no server, so this has to ship as data embedded in the HTML.
    A dense species x GO-term count matrix here is ~1200 x ~24000 cells --
    too big even gzipped as a plain typed-array dump. Two things shrink it,
    both lossless:

    1. Values always pack as uint16, even though a couple of cells in real
       data exceed 65535 -- those go in a tiny separate `overrides` list
       (flat GO-major index -> true value) applied after decoding, instead
       of doubling every one of the millions of *other* cells to uint32
       just to cover a handful of outliers.
    2. The matrix is ~70% zero, so it ships as a packed nonzero-bitmap plus
       a values array holding only the nonzero cells, concatenated and
       gzipped together -- smaller than gzipping the dense array directly,
       since gzip has to spend bits re-discovering the zero runs itself
       otherwise. The browser reverses this with the native
       DecompressionStream("gzip") API (no bundled inflate library needed),
       then rebuilds the dense typed array the rest of the JS expects by
       scattering values back in at the bitmap's set-bit positions.

    Layout is GO-major: all `len(species)` counts for go_ids[0], then all
    counts for go_ids[1], etc. -- so looking up one GO id's counts for
    every species is a single contiguous slice, not a strided gather.
    """
    go_ids = list(raw_full.columns)
    counts = raw_full.loc[species, go_ids].to_numpy()
    counts = np.clip(counts, 0, None)

    by_go = np.ascontiguousarray(counts.T)  # shape (n_go, n_species), GO-major
    flat = by_go.reshape(-1)

    UINT16_MAX = 65535
    overflow_positions = np.flatnonzero(flat > UINT16_MAX)
    overrides = [[int(p), int(flat[p])] for p in overflow_positions]

    flat_clipped = np.minimum(flat, UINT16_MAX).astype(np.uint16)
    nonzero_mask = flat_clipped != 0
    mask_bytes = np.packbits(nonzero_mask).tobytes()
    if len(mask_bytes) % 2:
        # Pad to an even length so the browser can lay a Uint16Array view
        # directly over the values that follow (typed-array byteOffsets
        # must be a multiple of the element size).
        mask_bytes += b"\x00"
    values_bytes = flat_clipped[nonzero_mask].tobytes()
    compressed = gzip.compress(mask_bytes + values_bytes, compresslevel=6)

    return {
        "go_ids": go_ids,
        "go_desc": {go_id: go_desc.get(go_id, "unknown") for go_id in go_ids},
        "n_species": len(species),
        "n_go": len(go_ids),
        "mask_bytes": len(mask_bytes),
        "overrides": overrides,
        "counts_gzip_b64": base64.b64encode(compressed).decode("ascii"),
    }


def decode_go_search_payload(go_search):
    """
    Python-side inverse of build_go_search_payload: reconstructs the raw
    species x GO-term counts matrix from the payload embedded in an already-
    generated general-PCA HTML page.

    Lets a follow-up analysis (e.g. go_group_diff.py) recover the exact
    counts a given plot was built from, without needing the original raw
    matrix file on disk -- the matrix behind a given HTML page may have come
    from a since-changed merge/filter step, so the page itself is the only
    reliably-consistent source once it exists.

    Returns (counts, go_ids): counts is an (n_species, n_go) uint32 ndarray,
    species-major, in the same row order as go_search's own species list
    (i.e. the order the caller's species records were built in); go_ids is
    the column label for each column.
    """
    go_ids = go_search["go_ids"]
    n_go = go_search["n_go"]
    n_species = go_search["n_species"]
    mask_bytes_len = go_search["mask_bytes"]

    raw = gzip.decompress(base64.b64decode(go_search["counts_gzip_b64"]))
    mask_part = raw[:mask_bytes_len]
    values_part = raw[mask_bytes_len:]

    total = n_go * n_species
    nonzero_mask = np.unpackbits(np.frombuffer(mask_part, dtype=np.uint8))[:total].astype(bool)
    values = np.frombuffer(values_part, dtype=np.uint16)

    flat = np.zeros(total, dtype=np.uint32)
    flat[nonzero_mask] = values
    for pos, val in go_search["overrides"]:
        flat[pos] = val

    by_go = flat.reshape(n_go, n_species)  # GO-major, matches the encode-side layout
    counts = np.ascontiguousarray(by_go.T)  # species-major (n_species, n_go)
    return counts, go_ids
