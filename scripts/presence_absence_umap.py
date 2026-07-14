#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general UMAP
embedding of GO term presence/absence across species -- no GO tree, no
illumination. UMAP counterpart of presence_absence_pca.py.

Same input matrix and preprocessing as presence_absence_pca.py (raw GO
counts, species x GO terms, rare-term filter, binarized to presence/
absence, StandardScaler), but the embedding itself is fit with UMAP
instead of TruncatedSVD. UMAP has no linear components_, so there is no
"top GO terms per axis" sidebar here -- see general_umap_common.py.
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers
from general_umap_common import (
    TEMPLATE_PATH,
    DEFAULT_IC_PATH,
    DATA_MARKER,
    TITLE_MARKER,
    rgb_to_hex,
    load_go_descriptions,
    build_go_search_payload,
    run_umap,
)


def run_umap_on_presence_absence(raw_df, n_neighbors, min_dist, metric, random_state):
    """
    Same rare-GO-term filter as presence_absence_pca.py (keep columns with
    raw count sum > 5 across species), binarized to presence/absence
    (count > 0 -> 1), StandardScaler, then UMAP instead of TruncatedSVD.

    Also returns each species' GO-term richness (how many of the retained
    columns it has any annotation for), for the tooltip.
    """
    umap_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    presence = (umap_input > 0).astype(float)

    scaler = StandardScaler()
    normalized = pd.DataFrame(scaler.fit_transform(presence.values), index=raw_df.index)

    embedding = run_umap(
        normalized,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    richness = presence.sum(axis=1)
    return embedding, richness, umap_input.shape[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive UMAP embedding of GO term presence/absence (no GO tree, no illumination)"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_umap_presence_absence.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", default="euclidean", help="UMAP distance metric (default: euclidean)")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP random_state, for reproducible layouts (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    taxon_dict = load_taxonomy(args.taxonomy)

    # Restrict to the requested taxa *before* fitting UMAP, not after -- same
    # rationale as presence_absence_pca.py's -t/--taxa: the embedding should
    # reflect variance only among the selected species, not a crop of a
    # global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    umap_df, richness, n_go_used = run_umap_on_presence_absence(
        raw_full, args.n_neighbors, args.min_dist, args.metric, args.random_state
    )
    # remove_outliers (shared with the PCA branch) hardcodes PC1/PC2 column
    # names -- rename around the call rather than reimplementing the same
    # percentile filter under a UMAP-specific name.
    umap_df = umap_df.rename(columns={"UMAP1": "PC1", "UMAP2": "PC2"})
    umap_df = remove_outliers(umap_df, low=5, high=95)
    umap_df = umap_df.rename(columns={"PC1": "UMAP1", "PC2": "UMAP2"})

    umap_df = umap_df.copy()
    umap_df["Group"] = umap_df.index.map(taxon_dict)
    umap_df = umap_df.dropna(subset=["Group"])

    species = list(umap_df.index)
    color_map = build_global_color_map(taxon_dict)

    species_records = [
        {
            "name": name,
            "umap1": float(umap_df.loc[name, "UMAP1"]),
            "umap2": float(umap_df.loc[name, "UMAP2"]),
            "group": umap_df.loc[name, "Group"],
            "go_terms_present": int(richness.get(name, 0)),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    go_desc = load_go_descriptions(args.ic_file)
    go_search = build_go_search_payload(raw_full, species, go_desc)

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "go_search": go_search,
        "meta": {
            "n_go_terms_used": int(n_go_used),
            "title": "General UMAP: GO term presence/absence",
            "mode_label": "presence/absence, not abundance -- UMAP embedding",
            "filename_base": "general_umap_presence_absence",
        },
    }

    template = TEMPLATE_PATH.read_text()
    title = payload["meta"]["title"]
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species, {n_go_used} GO columns used)")


if __name__ == "__main__":
    main()
