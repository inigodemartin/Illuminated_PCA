#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general UMAP
embedding of GO term relative abundance across species -- no GO tree, no
illumination. UMAP counterpart of general_pca_abundance.py.

Same idea as presence_absence_umap.py (same template, same live GO-term
search/illumination), but the embedding is fit on relative abundance
(count / Total_prots per species), reusing interactive_go_tree.py's
species-stats loading, instead of binarized presence/absence.
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers
from interactive_go_tree import load_species_stats
from general_umap_common import (
    TEMPLATE_PATH,
    DEFAULT_IC_PATH,
    DATA_MARKER,
    TITLE_MARKER,
    rgb_to_hex,
    load_go_ic_and_descriptions,
    build_go_search_payload,
    run_umap,
)


def run_umap_on_relative_abundance(raw_df, total_prots, n_neighbors, min_dist, metric, random_state):
    """
    Same rare-GO-term filter and relative-abundance conversion (count /
    Total_prots) as interactive_go_tree.run_pca_on_relative_abundance,
    StandardScaler, then UMAP instead of TruncatedSVD.
    """
    species = [s for s in raw_df.index if s in total_prots.index]
    raw_df = raw_df.loc[species]

    umap_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    total_prots_col = total_prots.loc[species].to_numpy(dtype="float64")[:, None]
    relative_values = umap_input.to_numpy(dtype="float64") / total_prots_col

    scaler = StandardScaler()
    normalized = pd.DataFrame(scaler.fit_transform(relative_values), index=species)

    embedding = run_umap(
        normalized,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return embedding, umap_input


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive UMAP embedding of GO term relative abundance (no GO tree, no illumination)"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--species-stats", required=True, help="TSV with a Species index and a Total_prots column")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_umap_abundance.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--ic-threshold", type=float, default=None,
                        help="Minimum IC to include a GO term in the embedding; GOs below this value are dropped from the matrix before fitting")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", default="euclidean", help="UMAP distance metric (default: euclidean)")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP random_state, for reproducible layouts (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)

    go_ic, go_desc_raw = load_go_ic_and_descriptions(args.ic_file)
    # Embed IC in the description string so it surfaces everywhere the
    # description is shown: GO search suggestions, tooltips, etc.
    go_desc = {
        go_id: f"{desc} (IC: {go_ic[go_id]:.2f})" if go_id in go_ic else desc
        for go_id, desc in go_desc_raw.items()
    }

    # Restrict to the requested taxa *before* fitting UMAP, not after -- same
    # rationale as presence_absence_umap.py's -t/--taxa: the embedding should
    # reflect variance only among the selected species, not a crop of a
    # global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    # Drop GO terms below the IC threshold before fitting so that overly
    # general terms (present in nearly all species, low information
    # content) don't dominate the embedding.
    n_absent_ic = sum(1 for c in raw_full.columns if c not in go_ic)
    if n_absent_ic:
        print(f"Warning: {n_absent_ic} GO terms in matrix have no IC value in {args.ic_file}")

    if args.ic_threshold is not None:
        n_before = raw_full.shape[1]
        raw_full = raw_full[[c for c in raw_full.columns if go_ic.get(c, 0.0) >= args.ic_threshold]]
        print(f"IC filter (≥ {args.ic_threshold}): kept {raw_full.shape[1]} / {n_before} GO terms")

    umap_df, umap_input = run_umap_on_relative_abundance(
        raw_full, total_prots, args.n_neighbors, args.min_dist, args.metric, args.random_state
    )
    n_go_used = umap_input.shape[1]
    # How many of the same GO columns the embedding was fit on each species
    # has any annotation for, for the tooltip.
    richness = (umap_input > 0).sum(axis=1)

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

    go_search = build_go_search_payload(raw_full, species, go_desc)

    title = "General UMAP: GO term relative abundance"
    mode_label = "relative abundance, not presence/absence -- UMAP embedding"
    if args.ic_threshold is not None:
        title += f" (IC ≥ {args.ic_threshold})"
        mode_label += f", IC ≥ {args.ic_threshold}"

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "go_search": go_search,
        "meta": {
            "n_go_terms_used": int(n_go_used),
            "title": title,
            "mode_label": mode_label,
            "filename_base": "general_umap_abundance",
        },
    }

    template = TEMPLATE_PATH.read_text()
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species, {n_go_used} GO columns used)")


if __name__ == "__main__":
    main()
