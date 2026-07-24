#!/usr/bin/env python3
"""
Generate a standalone interactive HTML explorer for a GO ancestor/descendant
tree: click a node to expand its illuminated PCA inline, hover a point to see
the species, its GO count and how that compares to its total protein count.

Reuses the ontology-tree resolution from go_tree_illuminated_pca.py and the
taxonomy/color helpers from illuminate_PCA.py; everything else (single-shot
PCA on a relative-abundance matrix, sparse per-node counts, HTML rendering)
is specific to this script.
"""

from pathlib import Path
import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers, assign_taxonomy_group
from go_tree_illuminated_pca import get_go_relations
from general_pca_common import (
    DEFAULT_IC_PATH,
    load_go_ic_and_descriptions,
    compute_species_contributions,
    top_loadings_by_pc,
    write_top_loadings_tsv,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "interactive_tree_template.html"
DATA_MARKER = "__INTERACTIVE_GO_TREE_DATA__"
TITLE_MARKER = "__INTERACTIVE_GO_TREE_TITLE__"


def load_ic_table(ic_file):
    """
    GO -> Information Content, from a headerless TSV
    (go_id, category, col3, col4, ic, description, trailing-tab).
    A few thousand GO ids appear twice with byte-identical rows; keep the
    first occurrence.
    """
    ic = {}
    with open(ic_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] in ic:
                continue
            try:
                ic[parts[0]] = float(parts[4])
            except ValueError:
                continue
    return ic


def load_species_stats(stats_file):
    """
    Species -> Total_prots (total protein count for that species).

    merged_species_stats.tsv has, for ~600 species, a second row that
    duplicates the species name with "-" placeholders in every column
    except IC_fan/IC_hom. Coerce to numeric (turning "-" into NaN) and keep
    the first valid value per species so those placeholder rows drop out.
    """
    stats = pd.read_csv(stats_file, sep="\t", index_col=0)
    total_prots = pd.to_numeric(stats["Total_prots"], errors="coerce").dropna()
    return total_prots[~total_prots.index.duplicated(keep="first")]


def filter_species_by_stats(raw_df, total_prots):
    """
    Restrict raw_df to species that have a valid Total_prots value, warning
    how many were dropped -- species present in the matrix but missing (or
    with a non-numeric placeholder) in --species-stats used to silently
    disappear from the PCA with no indication why.
    """
    species = [s for s in raw_df.index if s in total_prots.index]
    n_missing = raw_df.shape[0] - len(species)
    if n_missing:
        print(f"Warning: {n_missing} species in matrix have no Total_prots in species-stats — excluded from PCA")
    return raw_df.loc[species]


def run_pca_on_relative_abundance(raw_df, total_prots):
    """
    Drop rare GO columns, convert raw counts to a centered log-ratio (CLR)
    per species, then PCA via StandardScaler + TruncatedSVD. Computed once:
    the layout is identical for every GO node, so unlike the PNG pipeline
    this never needs to be recomputed per node.

    GO abundance is compositional (each species' counts are parts of a
    fixed total), so plain relative-abundance values fed into a Euclidean
    PCA produce spurious anticorrelations (Aitchison 1986). CLR maps each
    species row into the log-ratio space where Euclidean PCA is valid.
    CLR is scale-invariant (clr(x) == clr(c*x)), so total_prots is not
    needed here -- only used by callers for reporting.

    Also returns the per-GO-term loadings (model.components_, transposed so
    rows are GO ids): how much each retained GO column contributes to PC1/
    PC2, for the "most influential GO terms per component" report.
    """
    raw_df = filter_species_by_stats(raw_df, total_prots)
    species = list(raw_df.index)

    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    counts = pca_input.to_numpy(dtype="float64") + 1.0  # pseudo-count: log(0) is undefined
    log_counts = np.log(counts)
    clr_values = log_counts - log_counts.mean(axis=1, keepdims=True)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(clr_values)

    model = TruncatedSVD(n_components=2)
    components = model.fit_transform(normalized)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=species)
    loadings = pd.DataFrame(model.components_.T, columns=["PC1", "PC2"], index=pca_input.columns)
    normalized_df = pd.DataFrame(normalized, columns=pca_input.columns, index=species)
    return pca_df, model.explained_variance_ratio_, loadings, normalized_df


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def build_descendant_sum_cache(parent_to_children, own_vector, n_species):
    """
    Returns a function go_id -> per-species count vector summed over
    go_id + all its descendants, memoized across calls via an iterative
    post-order traversal (no recursion -- a real ontology's ancestor
    chains can run thousands of terms deep, which blew Python's
    recursion limit in an earlier, recursive version of this).

    Earlier version of this memoized full descendant-*id-set* per node
    instead of a count vector; for a deep/broad real ontology that's
    O(V^2) ids materialized (every node's set can be a sizeable fraction
    of the whole ontology) and exhausted memory. A fixed-size numpy
    vector per node is O(V * n_species) total instead, however big the
    ontology, and is exactly what's needed here anyway (we only ever sum
    counts, never inspect which ids contributed).

    Query-tree nodes (ancestors of a broad/shallow term especially, whose
    own descendant subtrees overlap heavily) used to each trigger an
    independent full BFS+sum; memoizing means every node's vector is
    computed once and reused, however many of the N query nodes need it.
    """
    memo = {}
    zeros = np.zeros(n_species, dtype="int64")

    def compute(root):
        if root in memo:
            return memo[root]

        stack = [root]
        on_path = set()
        while stack:
            node = stack[-1]
            if node in memo:
                stack.pop()
                continue
            if node in on_path:
                total = own_vector(node).copy()
                for child in parent_to_children.get(node, []):
                    total += memo.get(child, zeros)
                memo[node] = total
                on_path.discard(node)
                stack.pop()
                continue
            on_path.add(node)
            for child in parent_to_children.get(node, []):
                if child not in memo and child not in on_path:
                    stack.append(child)

        return memo[root]

    return compute


def compute_topological_levels(root, all_nodes, edges, plot_descendants):
    """
    Longest-path layering, the same property Graphviz's own rank assignment
    guarantees: every edge (parent, child) must have level(parent) >
    level(child). A GO term can have multiple parents at different
    distances from the root, so the shortest BFS distance (what
    get_go_relations returns) is not enough -- a node has to sit past the
    *longest* chain that reaches it, or some edge would point the wrong way.
    """
    children_of = defaultdict(list)
    parents_of = defaultdict(list)
    for parent, child in edges:
        children_of[parent].append(child)
        parents_of[child].append(parent)

    # Ancestors mode: walk from a node toward the root via its children
    # (closer to the root). Descendants mode: walk via its parents instead.
    neighbors = parents_of if plot_descendants else children_of

    # Iterative post-order (no recursion): a query term's ancestor chain
    # in a real ontology can run thousands of terms deep, which is enough
    # to blow Python's default recursion limit.
    levels = {}
    for root in all_nodes:
        if root in levels:
            continue
        stack = [root]
        on_path = set()
        while stack:
            node = stack[-1]
            if node in levels:
                stack.pop()
                continue
            if node in on_path:
                ups = neighbors.get(node, [])
                levels[node] = 1 + max((levels.get(n, 0) for n in ups), default=-1) if ups else 0
                on_path.discard(node)
                stack.pop()
                continue
            on_path.add(node)
            for n in neighbors.get(node, []):
                if n not in levels and n not in on_path:
                    stack.append(n)
    return levels


def build_tree(go_id, extra_nodes, go_desc, child_to_parents, parent_to_children, plot_descendants, ic_table):
    """
    Mirrors plot_go_ancestors/plot_go_descendants in go_tree_illuminated_pca.py,
    but emits plain data (nodes + edges) instead of a Graphviz diagram.
    """
    all_nodes = set(extra_nodes.keys()) | {go_id}

    edges = []
    if plot_descendants:
        for parent in sorted(all_nodes):
            for child in sorted(parent_to_children.get(parent, [])):
                if child in all_nodes:
                    edges.append([parent, child])
    else:
        for child in sorted(all_nodes):
            for parent in sorted(child_to_parents.get(child, [])):
                if parent in all_nodes:
                    edges.append([parent, child])

    levels = compute_topological_levels(go_id, all_nodes, edges, plot_descendants)

    nodes = [{
        "go_id": go_id,
        "description": go_desc.get(go_id, "unknown"),
        "level": levels[go_id],
        "is_root": True,
        "ic": ic_table.get(go_id),
    }]
    for node_id, info in extra_nodes.items():
        nodes.append({
            "go_id": node_id,
            "description": info["desc"],
            "level": levels[node_id],
            "is_root": False,
            "ic": ic_table.get(node_id),
        })

    return nodes, edges, all_nodes


def build_node_counts(node_ids, raw_full_df, species, parent_to_children, count_descendants):
    """
    {go_id: {species_index(str): raw_count}}, sparse (zero counts omitted).
    Always computed from the *full* raw matrix (not the rare-column-filtered
    one used for PCA), matching how illumination counts work today.
    """
    species_df = raw_full_df.loc[species]
    col_index = {go_id: i for i, go_id in enumerate(species_df.columns)}
    values = species_df.to_numpy(dtype="int64")
    n_species = len(species)
    zeros = np.zeros(n_species, dtype="int64")

    def own_vector(go_id):
        idx = col_index.get(go_id)
        return values[:, idx] if idx is not None else zeros

    sum_cache = build_descendant_sum_cache(parent_to_children, own_vector, n_species) if count_descendants else None

    counts = {}
    for go_id in node_ids:
        vector = sum_cache(go_id) if count_descendants else own_vector(go_id)
        sparse = {str(idx): int(value) for idx, value in enumerate(vector) if value > 0}
        counts[go_id] = sparse
    return counts


def _log(prev_t, message):
    """Print elapsed-since-last-stage time to stderr; returns the new checkpoint."""
    now = time.perf_counter()
    print(f"  [+{now - prev_t:6.1f}s] {message}", file=sys.stderr)
    return now


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive HTML GO ancestor/descendant tree with illuminated PCA")
    parser.add_argument("--go", "-g", required=True, help="Root GO ID")
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix (species x GO, TSV)")
    parser.add_argument("--species-stats", required=True, help="TSV with a Species index and a Total_prots column")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument("--obo", required=True, help="GO OBO file")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO Information Content TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--ic-threshold", type=float, default=None,
                        help="Minimum IC to include a GO term in the PCA; GOs below this value are dropped from the matrix before fitting")
    parser.add_argument("-t", "--taxa", nargs="*", default=None, help="Restrict to these taxonomic groups")
    parser.add_argument("-d", "--count_descendants", action="store_true", help="Sum counts over each node's own descendants too")
    parser.add_argument("-o", "--no_outliers", action="store_true", help="Robust (percentile-clipped) scaling instead of log scaling")
    parser.add_argument("-p", "--plot_descendants", action="store_true", help="Build the tree from descendants instead of ancestors")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--top-loadings-n", type=int, default=20,
                         help="Number of most-influential GO terms to report per PC (default: 20)")
    parser.add_argument("--loadings-output", default=None,
                         help="Top-loadings TSV path (default: alongside --output, with _top_loadings.tsv)")
    parser.add_argument("--outlier-percentile", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                         help="Drop species whose PC1 or PC2 falls outside this percentile range "
                              "(default: 0 100, i.e. no trimming). Pass e.g. '5 95' to trim.")
    return parser.parse_args()


def main():
    args = parse_args()
    t = time.perf_counter()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    t = _log(t, f"loaded matrix ({raw_full.shape[0]} species x {raw_full.shape[1]} GO columns)")
    total_prots = load_species_stats(args.species_stats)
    t = _log(t, f"loaded species stats ({len(total_prots)} species)")
    taxon_dict = load_taxonomy(args.taxonomy)
    t = _log(t, f"loaded taxonomy ({len(taxon_dict)} species)")

    go_ic_full, go_desc_full = load_go_ic_and_descriptions(args.ic_file)
    t = _log(t, f"loaded GO IC/description table ({len(go_ic_full)} GO terms)")

    raw_for_pca = raw_full
    if args.ic_threshold is not None:
        n_absent_ic = sum(1 for c in raw_for_pca.columns if c not in go_ic_full)
        if n_absent_ic:
            print(f"Warning: {n_absent_ic} GO terms in matrix have no IC value in {args.ic_file}", file=sys.stderr)
        n_before = raw_for_pca.shape[1]
        raw_for_pca = raw_for_pca[[c for c in raw_for_pca.columns if go_ic_full.get(c, 0.0) >= args.ic_threshold]]
        t = _log(t, f"IC filter (>= {args.ic_threshold}): kept {raw_for_pca.shape[1]} / {n_before} GO terms")

    pca_df, explained_variance, loadings, normalized_df = run_pca_on_relative_abundance(raw_for_pca, total_prots)
    outlier_low, outlier_high = args.outlier_percentile
    n_before_outliers = pca_df.shape[0]
    pca_df = remove_outliers(pca_df, low=outlier_low, high=outlier_high)
    n_dropped = n_before_outliers - pca_df.shape[0]
    if n_dropped:
        print(f"Outlier trim (percentile {outlier_low}-{outlier_high}): dropped {n_dropped} / {n_before_outliers} species", file=sys.stderr)
    t = _log(t, "ran PCA on relative abundance")

    pca_df = assign_taxonomy_group(pca_df, taxon_dict)
    if args.taxa:
        pca_df = pca_df[pca_df["Group"].isin(args.taxa)]

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)

    contributions = compute_species_contributions(
        normalized_df.loc[[s for s in species if s in normalized_df.index]],
        loadings,
        n=args.top_loadings_n,
    )
    t = _log(t, f"computed per-species PC contributions (top {args.top_loadings_n} per direction)")

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "group": pca_df.loc[name, "Group"],
            "total_prots": float(total_prots.get(name, 0.0)),
            "contributions": contributions.get(name, {}),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}
    t = _log(t, f"built species records ({len(species_records)} species)")

    ic_table = load_ic_table(args.ic_file)
    t = _log(t, f"loaded IC table ({len(ic_table)} GO terms)")

    ancestors, descendants, go_desc, child_to_parents, parent_to_children = get_go_relations(args.go, args.obo)
    t = _log(t, f"parsed OBO and resolved ancestors/descendants ({len(go_desc)} GO terms in ontology)")
    extra_nodes = descendants if args.plot_descendants else ancestors
    nodes, edges, all_node_ids = build_tree(
        args.go, extra_nodes, go_desc, child_to_parents, parent_to_children, args.plot_descendants, ic_table
    )
    t = _log(t, f"built tree ({len(nodes)} nodes, {len(edges)} edges)")

    counts = build_node_counts(all_node_ids, raw_full, species, parent_to_children, args.count_descendants)
    t = _log(t, f"computed per-node GO counts ({len(counts)} nodes, count_descendants={args.count_descendants})")

    top_loadings = top_loadings_by_pc(loadings, go_desc, args.top_loadings_n)
    t = _log(t, f"ranked top {args.top_loadings_n} GO terms per PC by loading")

    # Descriptions for whichever GO ids show up in the per-species
    # contributions modal -- not tied to the ontology subtree currently on
    # screen, since a species' top-contributing terms can be anywhere in
    # the matrix. Kept to just the referenced ids (not the full IC table)
    # so the payload stays small.
    contrib_go_ids = {
        entry["go_id"]
        for sp in contributions.values()
        for pc_data in sp.values()
        for entry in pc_data["positive"] + pc_data["negative"]
    }
    contrib_go_desc = {go_id: go_desc_full.get(go_id, "unknown") for go_id in contrib_go_ids}

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "tree": {"nodes": nodes, "edges": edges},
        "counts": counts,
        "top_loadings": top_loadings,
        "contrib_go_desc": contrib_go_desc,
        "meta": {
            "root": args.go,
            "mode": "descendants" if args.plot_descendants else "ancestors",
            "count_descendants": bool(args.count_descendants),
            "no_outliers": bool(args.no_outliers),
            "explained_variance": [float(v) for v in explained_variance],
            "ic_threshold": args.ic_threshold,
        },
    }

    template = TEMPLATE_PATH.read_text()
    title = f"Interactive GO tree: {args.go} ({go_desc.get(args.go, 'unknown')})"
    if args.ic_threshold is not None:
        title += f" (IC >= {args.ic_threshold})"
    title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title_safe).replace(DATA_MARKER, data_json)

    output_path = args.output or f"interactive_{args.go.replace(':', '_')}_{'descendants' if args.plot_descendants else 'ancestors'}.html"
    Path(output_path).write_text(html)
    _log(t, f"wrote {output_path}")

    loadings_output = args.loadings_output or f"{Path(output_path).with_suffix('')}_top_loadings.tsv"
    write_top_loadings_tsv(top_loadings, loadings_output)
    _log(t, f"wrote {loadings_output}")
    print(f"Wrote {output_path} ({len(nodes)} nodes, {len(species_records)} species)")
    print(f"Wrote {loadings_output} (top {args.top_loadings_n} GO terms per PC)")


if __name__ == "__main__":
    main()
