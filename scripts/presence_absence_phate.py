#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general PHATE
embedding of GO term presence/absence across species -- no GO tree, no
illumination. PHATE counterpart of presence_absence_pca.py /
presence_absence_umap.py.

Same input matrix and preprocessing as presence_absence_umap.py (raw GO
counts, species x GO terms, rare-term filter, binarized to presence/
absence, StandardScaler), but the embedding itself is fit with PHATE
instead of UMAP/TruncatedSVD -- PHATE preserves long-range/global
structure that UMAP's purely local neighborhood graph does not. PHATE has
no linear components_ either, so there is no "top GO terms per axis"
sidebar here -- see general_phate_common.py.
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers
from general_phate_common import (
    TEMPLATE_PATH,
    DEFAULT_IC_PATH,
    DATA_MARKER,
    TITLE_MARKER,
    rgb_to_hex,
    load_go_descriptions,
    build_go_search_payload,
    run_phate,
)


def _t_value(s):
    if s == "auto":
        return s
    return int(s)


def run_phate_on_presence_absence(raw_df, knn, decay, t, metric, random_state):
    """
    Same rare-GO-term filter as presence_absence_pca.py/presence_absence_umap.py
    (keep columns with raw count sum > 5 across species), binarized to
    presence/absence (count > 0 -> 1), StandardScaler, then PHATE instead
    of UMAP/TruncatedSVD.

    Also returns each species' GO-term richness (how many of the retained
    columns it has any annotation for), for the tooltip.
    """
    phate_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    presence = (phate_input > 0).astype(float)

    scaler = StandardScaler()
    normalized = pd.DataFrame(scaler.fit_transform(presence.values), index=raw_df.index)

    embedding = run_phate(
        normalized,
        knn=knn,
        decay=decay,
        t=t,
        metric=metric,
        random_state=random_state,
    )
    richness = presence.sum(axis=1)
    return embedding, richness, phate_input.shape[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive PHATE embedding of GO term presence/absence (no GO tree, no illumination)"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_phate_presence_absence.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--knn", type=int, default=5, help="PHATE knn -- neighborhood size for the diffusion graph (default: 5)")
    parser.add_argument("--decay", type=int, default=40, help="PHATE decay -- kernel bandwidth decay rate (default: 40)")
    parser.add_argument("--diffusion-t", dest="diffusion_t", type=_t_value, default="auto",
                         help="PHATE diffusion time t: 'auto' or an integer -- higher values emphasize more global structure (default: auto)")
    parser.add_argument("--metric", default="euclidean", help="PHATE knn distance metric (default: euclidean)")
    parser.add_argument("--random-state", type=int, default=42, help="PHATE random_state, for reproducible layouts (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    taxon_dict = load_taxonomy(args.taxonomy)

    # Restrict to the requested taxa *before* fitting PHATE, not after -- same
    # rationale as presence_absence_pca.py's -t/--taxa: the embedding should
    # reflect variance only among the selected species, not a crop of a
    # global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    phate_df, richness, n_go_used = run_phate_on_presence_absence(
        raw_full, args.knn, args.decay, args.diffusion_t, args.metric, args.random_state
    )
    # remove_outliers (shared with the PCA branch) hardcodes PC1/PC2 column
    # names -- rename around the call rather than reimplementing the same
    # percentile filter under a PHATE-specific name.
    phate_df = phate_df.rename(columns={"PHATE1": "PC1", "PHATE2": "PC2"})
    phate_df = remove_outliers(phate_df, low=5, high=95)
    phate_df = phate_df.rename(columns={"PC1": "PHATE1", "PC2": "PHATE2"})

    phate_df = phate_df.copy()
    phate_df["Group"] = phate_df.index.map(taxon_dict)
    phate_df = phate_df.dropna(subset=["Group"])

    species = list(phate_df.index)
    color_map = build_global_color_map(taxon_dict)

    species_records = [
        {
            "name": name,
            "phate1": float(phate_df.loc[name, "PHATE1"]),
            "phate2": float(phate_df.loc[name, "PHATE2"]),
            "group": phate_df.loc[name, "Group"],
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
            "title": "General PHATE: GO term presence/absence",
            "mode_label": "presence/absence, not abundance -- PHATE embedding",
            "filename_base": "general_phate_presence_absence",
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
