#!/usr/bin/env python3
"""
Shared helpers for the standalone-HTML general PCA scripts (presence_absence_pca.py,
general_pca_abundance.py): both render the same template and need the same GO
search/illumination payload, just fit on different PCA inputs.
"""

from pathlib import Path
import base64
import gzip

import numpy as np
import pandas as pd

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_pca_template.html"
DEFAULT_IC_PATH = Path(__file__).parent.parent / "All_GOs_ic.tsv"
DATA_MARKER = "__GENERAL_PCA_DATA__"
TITLE_MARKER = "__GENERAL_PCA_TITLE__"


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def load_go_descriptions(ic_file):
    """
    GO -> description, from the same bundled headerless TSV used for IC
    lookups elsewhere in this project (go_id, category, col3, col4, ic,
    description, trailing-tab). Only the description column is needed
    here, so this doesn't require a full OBO file.
    """
    desc = {}
    with open(ic_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[0] in desc:
                continue
            desc[parts[0]] = parts[5]
    return desc


def load_go_ic(ic_file):
    """GO id -> IC value (float), from the same headerless TSV (column 4)."""
    ic = {}
    with open(ic_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] in ic:
                continue
            try:
                ic[parts[0]] = float(parts[4])
            except ValueError:
                pass
    return ic


def top_loadings_by_pc(loadings, go_desc, n):
    """
    For each PC, the n GO terms with the largest |loading| -- the GO terms
    whose value most drives that axis, in either direction (sign kept).
    """
    result = {}
    for pc in loadings.columns:
        ranked = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
        top = ranked.head(n)
        result[pc] = [
            {"go_id": go_id, "description": go_desc.get(go_id, "unknown"), "loading": float(value)}
            for go_id, value in top.items()
        ]
    return result


def write_top_loadings_tsv(top_loadings, output_path):
    rows = []
    for pc, entries in top_loadings.items():
        for rank, entry in enumerate(entries, start=1):
            rows.append({
                "PC": pc,
                "Rank": rank,
                "GO_id": entry["go_id"],
                "Description": entry["description"],
                "Loading": entry["loading"],
            })
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


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
    A dense species x GO-term count matrix here is ~1200 x ~24000 cells,
    too big as JSON (~12M non-zero entries -- sparse wouldn't meaningfully
    shrink it, the matrix is ~33% dense). Instead: pack counts as a flat
    typed-array byte buffer (uint16, or uint32 if some count overflows
    that), gzip it, and base64-encode the gzipped bytes -- gzip alone gets
    this down roughly 5x, base64 adds ~33% back, netting a fraction of the
    raw size. The browser reverses this with the native
    DecompressionStream("gzip") API (no bundled inflate library needed).

    Layout is GO-major: all `len(species)` counts for go_ids[0], then all
    counts for go_ids[1], etc. -- so looking up one GO id's counts for
    every species is a single contiguous slice, not a strided gather.
    """
    go_ids = list(raw_full.columns)
    counts = raw_full.loc[species, go_ids].to_numpy()

    max_count = int(counts.max()) if counts.size else 0
    dtype = np.uint16 if max_count <= 65535 else np.uint32

    by_go = np.ascontiguousarray(counts.astype(dtype).T)  # shape (n_go, n_species)
    compressed = gzip.compress(by_go.tobytes(), compresslevel=9)

    return {
        "go_ids": go_ids,
        "go_desc": {go_id: go_desc.get(go_id, "unknown") for go_id in go_ids},
        "n_species": len(species),
        "bytes_per_value": int(np.dtype(dtype).itemsize),
        "counts_gzip_b64": base64.b64encode(compressed).decode("ascii"),
    }
