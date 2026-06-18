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
from collections import defaultdict, deque

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers
from go_tree_illuminated_pca import get_go_relations

TEMPLATE_PATH = Path(__file__).parent / "templates" / "interactive_tree_template.html"
DATA_MARKER = "__INTERACTIVE_GO_TREE_DATA__"
TITLE_MARKER = "__INTERACTIVE_GO_TREE_TITLE__"


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


def run_pca_on_relative_abundance(raw_df, total_prots):
    """
    Drop rare GO columns, convert raw counts to relative abundance
    (count / Total_prots per species), then PCA via StandardScaler +
    TruncatedSVD. Computed once: the layout is identical for every GO node,
    so unlike the PNG pipeline this never needs to be recomputed per node.
    """
    species = [s for s in raw_df.index if s in total_prots.index]
    raw_df = raw_df.loc[species]

    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    relative = pca_input.div(total_prots.loc[species], axis=0)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(relative.values)

    model = TruncatedSVD(n_components=2)
    components = model.fit_transform(normalized)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=species)
    return pca_df, model.explained_variance_ratio_


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def all_descendants_including_self(go_id, parent_to_children):
    """BFS over the full ontology's parent->children map."""
    out = {go_id}
    queue = deque([go_id])
    while queue:
        current = queue.popleft()
        for child in parent_to_children.get(current, []):
            if child not in out:
                out.add(child)
                queue.append(child)
    return out


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

    levels = {}

    def recurse(node):
        if node in levels:
            return levels[node]
        ups = neighbors.get(node, [])
        levels[node] = 1 + max(recurse(n) for n in ups) if ups else 0
        return levels[node]

    for node in all_nodes:
        recurse(node)
    return levels


def build_tree(go_id, extra_nodes, go_desc, child_to_parents, parent_to_children, plot_descendants):
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

    nodes = [{"go_id": go_id, "description": go_desc.get(go_id, "unknown"), "level": levels[go_id], "is_root": True}]
    for node_id, info in extra_nodes.items():
        nodes.append({"go_id": node_id, "description": info["desc"], "level": levels[node_id], "is_root": False})

    return nodes, edges, all_nodes


def build_node_counts(node_ids, raw_full_df, species, parent_to_children, count_descendants):
    """
    {go_id: {species_index(str): raw_count}}, sparse (zero counts omitted).
    Always computed from the *full* raw matrix (not the rare-column-filtered
    one used for PCA), matching how illumination counts work today.
    """
    counts = {}
    for go_id in node_ids:
        targets = all_descendants_including_self(go_id, parent_to_children) if count_descendants else {go_id}
        valid_cols = [g for g in targets if g in raw_full_df.columns]

        sparse = {}
        if valid_cols:
            summed = raw_full_df.loc[species, valid_cols].sum(axis=1)
            for idx, value in enumerate(summed.values):
                if value > 0:
                    sparse[str(idx)] = int(value)
        counts[go_id] = sparse
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive HTML GO ancestor/descendant tree with illuminated PCA")
    parser.add_argument("--go", "-g", required=True, help="Root GO ID")
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix (species x GO, TSV)")
    parser.add_argument("--species-stats", required=True, help="TSV with a Species index and a Total_prots column")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument("--obo", required=True, help="GO OBO file")
    parser.add_argument("-t", "--taxa", nargs="*", default=None, help="Restrict to these taxonomic groups")
    parser.add_argument("-d", "--count_descendants", action="store_true", help="Sum counts over each node's own descendants too")
    parser.add_argument("-o", "--no_outliers", action="store_true", help="Robust (percentile-clipped) scaling instead of log scaling")
    parser.add_argument("-p", "--plot_descendants", action="store_true", help="Build the tree from descendants instead of ancestors")
    parser.add_argument("--output", default=None, help="Output HTML path")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)

    pca_df, explained_variance = run_pca_on_relative_abundance(raw_full, total_prots)
    pca_df = remove_outliers(pca_df, low=5, high=95)

    pca_df = pca_df.copy()
    pca_df["Group"] = pca_df.index.map(taxon_dict)
    pca_df = pca_df.dropna(subset=["Group"])
    if args.taxa:
        pca_df = pca_df[pca_df["Group"].isin(args.taxa)]

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "group": pca_df.loc[name, "Group"],
            "total_prots": float(total_prots.get(name, 0.0)),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    ancestors, descendants, go_desc, child_to_parents, parent_to_children = get_go_relations(args.go, args.obo)
    extra_nodes = descendants if args.plot_descendants else ancestors
    nodes, edges, all_node_ids = build_tree(
        args.go, extra_nodes, go_desc, child_to_parents, parent_to_children, args.plot_descendants
    )

    counts = build_node_counts(all_node_ids, raw_full, species, parent_to_children, args.count_descendants)

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "tree": {"nodes": nodes, "edges": edges},
        "counts": counts,
        "meta": {
            "root": args.go,
            "mode": "descendants" if args.plot_descendants else "ancestors",
            "count_descendants": bool(args.count_descendants),
            "no_outliers": bool(args.no_outliers),
            "explained_variance": [float(v) for v in explained_variance],
        },
    }

    template = TEMPLATE_PATH.read_text()
    title = f"Interactive GO tree: {args.go} ({go_desc.get(args.go, 'unknown')})"
    title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title_safe).replace(DATA_MARKER, data_json)

    output_path = args.output or f"interactive_{args.go.replace(':', '_')}_{'descendants' if args.plot_descendants else 'ancestors'}.html"
    Path(output_path).write_text(html)
    print(f"Wrote {output_path} ({len(nodes)} nodes, {len(species_records)} species)")


if __name__ == "__main__":
    main()
