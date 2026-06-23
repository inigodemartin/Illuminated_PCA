#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general PCA of
GO term presence/absence across species -- no GO tree, no illumination.

Same input matrix as interactive_go_tree.py (raw GO counts, species x GO
terms), but each retained GO column is binarized (count > 0 -> 1) instead
of converted to relative abundance, since presence/absence doesn't need
the Total_prots normalization the abundance PCA uses.
"""

from pathlib import Path
import argparse
import base64
import gzip
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_pca_template.html"
DEFAULT_IC_PATH = Path(__file__).parent.parent / "data" / "All_GOs_ic.tsv"
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


def run_pca_on_presence_absence(raw_df):
    """
    Same rare-GO-term filter as the relative-abundance PCA (keep columns
    with raw count sum > 5 across species), but binarize the retained
    columns to presence/absence (count > 0 -> 1) instead of dividing by
    Total_prots, before StandardScaler + TruncatedSVD.

    Also returns each species' GO-term richness (how many of the retained
    columns it has any annotation for) for the tooltip, and the per-GO-term
    loadings (model.components_, transposed so rows are GO ids): how much
    each retained GO column contributes to PC1/PC2.
    """
    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    presence = (pca_input > 0).astype(float)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(presence.values)

    model = TruncatedSVD(n_components=2)
    components = model.fit_transform(normalized)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=raw_df.index)
    richness = presence.sum(axis=1)
    loadings = pd.DataFrame(model.components_.T, columns=["PC1", "PC2"], index=pca_input.columns)
    return pca_df, model.explained_variance_ratio_, richness, loadings


def top_loadings_by_pc(loadings, go_desc, n):
    """
    For each PC, the n GO terms with the largest |loading| -- the GO terms
    whose presence/absence most drives that axis, in either direction
    (sign kept).
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive PCA of GO term presence/absence (no GO tree, no illumination)"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_pca_presence_absence.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--top-loadings-n", type=int, default=15,
                         help="Number of most-influential GO terms to report per PC (default: 15)")
    parser.add_argument("--loadings-output", default=None,
                         help="Top-loadings TSV path (default: alongside --output, with _top_loadings.tsv)")
    parser.add_argument("-o", "--no_outliers", action="store_true",
                         help="Robust (percentile-clipped) scaling instead of log scaling when illuminating a searched GO term")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    taxon_dict = load_taxonomy(args.taxonomy)

    # Restrict to the requested taxa *before* running PCA, not after: the
    # whole point of -t/--taxa is to compute the PCA only from variance
    # among those species, not to compute it on everyone and crop the plot
    # to a sub-region of the same global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    pca_df, explained_variance, richness, loadings = run_pca_on_presence_absence(raw_full)
    n_go_used = loadings.shape[0]
    pca_df = remove_outliers(pca_df, low=5, high=95)

    pca_df = pca_df.copy()
    pca_df["Group"] = pca_df.index.map(taxon_dict)
    pca_df = pca_df.dropna(subset=["Group"])

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "group": pca_df.loc[name, "Group"],
            "go_terms_present": int(richness.get(name, 0)),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    go_desc = load_go_descriptions(args.ic_file)
    top_loadings = top_loadings_by_pc(loadings, go_desc, args.top_loadings_n)
    go_search = build_go_search_payload(raw_full, species, go_desc)

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "top_loadings": top_loadings,
        "go_search": go_search,
        "meta": {
            "n_go_terms_used": int(n_go_used),
            "explained_variance": [float(v) for v in explained_variance],
            "no_outliers": bool(args.no_outliers),
        },
    }

    template = TEMPLATE_PATH.read_text()
    title = "General PCA: GO term presence/absence"
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species, {n_go_used} GO columns used)")

    loadings_output = args.loadings_output or f"{Path(args.output).with_suffix('')}_top_loadings.tsv"
    write_top_loadings_tsv(top_loadings, loadings_output)
    print(f"Wrote {loadings_output} (top {args.top_loadings_n} GO terms per PC)")


if __name__ == "__main__":
    main()
