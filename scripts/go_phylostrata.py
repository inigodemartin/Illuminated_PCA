#!/usr/bin/env python3
"""
Phylostratigraphy of GO terms: instead of embedding GO terms by cross-species
abundance PATTERN (go_space_common / explore_go_space_pca.py) -- which colors
by ontology category (BP/MF/CC) and, per the user, doesn't show anything
clear because category has no relationship to that embedding's structure --
this asks a different question directly: at what point in evolution did each
GO term likely originate?

Backbone tree (hand-built from the 12 distinct values in the Group column of
merged_taxons_belen.tsv -- there is no branch-length species tree available
locally, only this coarse grouping):

    Root
    +-- Asgard                              (archaea, outgroup)
    +-- Eukaryota
        +-- Protists                        (paraphyletic grab-bag; can't
        |                                     resolve further with this data)
        +-- Opisthokonta
        |   +-- Fungi
        |   +-- Metazoa
        +-- Archaeplastida
            +-- Glaucophyta
            +-- Rhodophyta
            +-- Viridiplantae
                +-- chlorophyta
                +-- Streptophyta
                    +-- bryophytes
                    +-- Tracheophyta
                        +-- lycophytes
                        +-- Euphyllophyta
                            +-- pteridophyte
                            +-- Spermatophyta
                                +-- gymnosperms
                                +-- angiosperms

This is a standard, defensible consensus topology, not a fitted tree -- good
enough to place GO-term origins at the right resolution (which named group
first has it), not to make claims about branch lengths or exact node ages.

Method (single origin + independent losses -- i.e. Dollo parsimony, the
standard assumption in phylostratigraphy/gene-age dating):
  1. A GO term is "present" in a leaf group if enough of that group's species
     have a nonzero count for it -- not just one, since FANTASIA's
     embedding-based predictions do throw the occasional false positive in an
     otherwise-negative clade. Threshold: >= max(--min-count, ceil(--min-frac
     * group size)) species positive.
  2. The GO term's origin node = the most recent common ancestor (MRCA) of
     every leaf group where it's present. A term present only in Metazoa
     origin-dates to the Metazoa leaf itself (animal-specific); a term
     present in both Metazoa and angiosperms dates to Eukaryota (their MRCA)
     under the single-origin assumption, even though independent convergent
     origin is also possible -- phylostratigraphy conventionally picks MRCA.
  3. Terms present in zero leaf groups (too rare/noisy everywhere to clear
     the threshold) are excluded from the tree and reported separately.

Output: a node-link tree plot with marker size/color intensity by count of
GO terms originating there (the "burst of functional innovation" at each
evolutionary transition), a stacked bar panel breaking those counts down by
GO category per node, and a full TSV report for follow-up (e.g. what are the
Metazoa-specific novel functions?).
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from general_pca_common import DEFAULT_IC_PATH, load_go_ic_and_descriptions
from illuminate_PCA import load_taxonomy
from interactive_go_tree import load_species_stats, filter_species_by_stats

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
})

CATEGORY_COLOR = {
    "biological_process": "#4C9BE8",
    "molecular_function": "#F5A623",
    "cellular_component": "#888888",
    "unknown": "#000000",
}

TREE_EDGES = [
    ("Root", "Asgard"),
    ("Root", "Eukaryota"),
    ("Eukaryota", "Protists"),
    ("Eukaryota", "Opisthokonta"),
    ("Opisthokonta", "Fungi"),
    ("Opisthokonta", "Metazoa"),
    ("Eukaryota", "Archaeplastida"),
    ("Archaeplastida", "Glaucophyta"),
    ("Archaeplastida", "Rhodophyta"),
    ("Archaeplastida", "Viridiplantae"),
    ("Viridiplantae", "chlorophyta"),
    ("Viridiplantae", "Streptophyta"),
    ("Streptophyta", "bryophytes"),
    ("Streptophyta", "Tracheophyta"),
    ("Tracheophyta", "lycophytes"),
    ("Tracheophyta", "Euphyllophyta"),
    ("Euphyllophyta", "pteridophyte"),
    ("Euphyllophyta", "Spermatophyta"),
    ("Spermatophyta", "gymnosperms"),
    ("Spermatophyta", "angiosperms"),
]
LEAF_GROUPS = [c for _, c in TREE_EDGES if c not in {p for p, _ in TREE_EDGES}]


def build_tree():
    parent, children = {}, defaultdict(list)
    for p, c in TREE_EDGES:
        parent[c] = p
        children[p].append(c)

    depth = {"Root": 0}
    order = ["Root"]
    i = 0
    while i < len(order):
        node = order[i]
        i += 1
        for ch in children.get(node, []):
            depth[ch] = depth[node] + 1
            order.append(ch)

    ancestors = {"Root": ["Root"]}
    for node in order[1:]:
        ancestors[node] = ancestors[parent[node]] + [node]

    return parent, dict(children), depth, ancestors, order


def mrca(present_groups, ancestors, depth):
    common = set(ancestors[present_groups[0]])
    for g in present_groups[1:]:
        common &= set(ancestors[g])
    return max(common, key=lambda n: depth[n])


def subtree_species_counts(children, group_sizes):
    """Species count backing each node: its own size if a leaf, otherwise the sum
    across every leaf descendant. Needed to normalize origination counts -- Fungi
    (849 species) and Metazoa (954) will rack up more raw originations than
    Glaucophyta (3) or pteridophyte (2) purely from sampling more species, not
    necessarily from being more evolutionarily innovative."""
    totals = {}

    def recurse(n):
        ch = children.get(n, [])
        if not ch:
            totals[n] = group_sizes.get(n, 0)
            return totals[n]
        totals[n] = sum(recurse(c) for c in ch)
        return totals[n]

    recurse("Root")
    return totals


def layout_y(node, children):
    """Post-order layout: leaves get sequential y; internal nodes sit at the
    mean of their children's y, the standard dendrogram placement."""
    ypos = {}
    next_y = [0]

    def recurse(n):
        ch = children.get(n, [])
        if not ch:
            ypos[n] = next_y[0]
            next_y[0] += 1
            return ypos[n]
        ys = [recurse(c) for c in ch]
        ypos[n] = sum(ys) / len(ys)
        return ypos[n]

    recurse(node)
    return ypos


def compute_presence(matrix_path, species_stats_path, taxon_path, min_frac, min_count):
    header_cols = pd.read_csv(matrix_path, sep="\t", nrows=0).columns
    dtype_map = {c: "float32" for c in header_cols[1:]}
    raw_df = pd.read_csv(matrix_path, sep="\t", index_col=0, dtype=dtype_map).fillna(0)
    total_prots = load_species_stats(species_stats_path)
    raw_df = filter_species_by_stats(raw_df, total_prots)

    taxon_dict = load_taxonomy(taxon_path)
    groups = raw_df.index.map(taxon_dict)
    mask = groups.isin(LEAF_GROUPS)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"Warning: {n_dropped} species have no/unrecognized Group -- excluded from phylostratigraphy")
    raw_df = raw_df.loc[mask]
    groups = groups[mask]

    go_ids = list(raw_df.columns)
    presence = raw_df.to_numpy() > 0  # species x go, bool
    del raw_df

    present_in_group = {}
    group_sizes = {}
    for g in LEAF_GROUPS:
        rows = np.asarray(groups == g)
        size = int(rows.sum())
        group_sizes[g] = size
        if size == 0:
            present_in_group[g] = np.zeros(len(go_ids), dtype=bool)
            continue
        counts = presence[rows].sum(axis=0)
        threshold = max(min_count, int(np.ceil(min_frac * size)))
        present_in_group[g] = counts >= threshold

    return go_ids, present_in_group, group_sizes


def assign_origins(go_ids, present_in_group, ancestors, depth):
    origin_by_go = {}
    unresolved = []
    cache = {}
    for j, go_id in enumerate(go_ids):
        present = tuple(g for g in LEAF_GROUPS if present_in_group[g][j])
        if not present:
            unresolved.append(go_id)
            continue
        if present not in cache:
            cache[present] = mrca(present, ancestors, depth)
        origin_by_go[go_id] = (cache[present], present)
    return origin_by_go, unresolved


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", "-m", default="merged_PCA_belen_fantasia.tsv",
                     help="Raw GO counts matrix, species x GO terms (TSV)")
    ap.add_argument("--species-stats", default="merged_species_stats_belen.tsv",
                     help="TSV with a Species index and a Total_prots column")
    ap.add_argument("--taxonomy", default="merged_taxons_belen.tsv",
                     help="TSV with Species and Group columns")
    ap.add_argument("--ic-file", default=str(DEFAULT_IC_PATH),
                     help="GO id -> category/IC/description TSV (default: bundled data/All_GOs_ic.tsv)")
    ap.add_argument("--min-frac", type=float, default=0.02,
                     help="A GO term counts as 'present' in a group if this fraction of the group's "
                          "species have it (default: 0.02 = 2%%)")
    ap.add_argument("--min-count", type=int, default=2,
                     help="...or at least this many species, whichever threshold is larger (default: 2)")
    ap.add_argument("--top-n", type=int, default=10,
                     help="GO terms to print per highlighted node, ranked by IC (default: 10)")
    ap.add_argument("--output", default="go_phylostrata.png", help="Output plot path")
    ap.add_argument("--report", default="go_phylostrata_report.tsv", help="Output per-GO-term TSV report path")
    ap.add_argument("--format", default="png", help="Plot format(s), comma-separated (default: png)")
    return ap.parse_args()


def main():
    args = parse_args()
    plot_formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    print(f"Loading {args.matrix} ...")
    go_ids, present_in_group, group_sizes = compute_presence(
        args.matrix, args.species_stats, args.taxonomy, args.min_frac, args.min_count)
    print(f"{len(go_ids)} GO terms x {len(LEAF_GROUPS)} groups. Group sizes (species surviving filters): {group_sizes}")

    parent, children, depth, ancestors, order = build_tree()
    origin_by_go, unresolved = assign_origins(go_ids, present_in_group, ancestors, depth)
    print(f"{len(origin_by_go)} GO terms placed on the tree, {len(unresolved)} unresolved "
          f"(didn't clear the presence threshold in any group)")

    go_ic, go_desc = load_go_ic_and_descriptions(args.ic_file)
    category_by_id = {}
    with open(args.ic_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] not in category_by_id:
                category_by_id[parts[0]] = parts[1]

    # --- per-node counts, overall and by category ---
    node_counts = Counter()
    node_category_counts = defaultdict(Counter)
    for go_id, (node, _present) in origin_by_go.items():
        node_counts[node] += 1
        node_category_counts[node][category_by_id.get(go_id, "unknown")] += 1

    # --- normalize by sampling effort: a clade with more species surveyed racks up
    # more raw originations for free (more chances to clear the presence threshold),
    # regardless of how evolutionarily innovative it actually is -- Fungi (849
    # species) vs Glaucophyta (3) isn't a fair comparison on raw counts alone ---
    node_species = subtree_species_counts(children, group_sizes)
    node_rate = {n: node_counts[n] / node_species[n] for n in node_counts if node_species.get(n, 0) > 0}

    # --- TSV report ---
    with open(args.report, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["GO_id", "description", "category", "IC", "origin_node", "present_groups",
                          "n_groups_present", "origin_node_n_species", "origin_node_rate_per_species"])
        for go_id in go_ids:
            if go_id not in origin_by_go:
                continue
            node, present = origin_by_go[go_id]
            writer.writerow([go_id, go_desc.get(go_id, "unknown"), category_by_id.get(go_id, "unknown"),
                              round(go_ic[go_id], 2) if go_id in go_ic else "", node,
                              ",".join(present), len(present), node_species.get(node, ""),
                              round(node_rate.get(node, 0), 4)])
    print(f"Wrote {args.report}")

    # --- plot 1: node-link tree, marker size/color by origination count ---
    Y_SCALE = 2.4  # spreads sibling leaves apart enough that adjacent labels don't collide
    ypos = {n: y * Y_SCALE for n, y in layout_y("Root", children).items()}
    xpos = depth

    fig = plt.figure(figsize=(13, 21))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.4, 1.1, 1.1, 1.1], hspace=0.6)
    ax_tree = fig.add_subplot(gs[0])
    ax_total = fig.add_subplot(gs[1])
    ax_rate = fig.add_subplot(gs[2])
    ax_frac = fig.add_subplot(gs[3])

    for p, c in TREE_EDGES:
        ax_tree.plot([xpos[p], xpos[c]], [ypos[p], ypos[c]], color="#BBBBBB", lw=1.2, zorder=1)

    # scatter's `s` is marker AREA (points^2) -- area must scale LINEARLY with count
    # for circle size to be an honest encoding of magnitude (area doubles iff count
    # doubles). A previous version applied an extra sqrt on top of this ratio, which
    # collapsed the effective encoding to radius ~ count**0.25: e.g. Root (7,828) and
    # Metazoa (6,086), a 30% difference in count, differed by only ~6% in radius --
    # real differences were nearly invisible. Color intensity can still use a mild
    # compressive (sqrt) scale for legibility since color has no such area convention.
    max_count = max(node_counts.values()) if node_counts else 1
    S_MIN, S_MAX = 30, 1400
    cmap = plt.cm.Blues
    # leaves alternate their label above/below the marker (by y-rank parity) so two
    # leaves that land close together in y never stack their text on top of each other
    leaf_rank = {n: i for i, n in enumerate(sorted(LEAF_GROUPS, key=lambda n: ypos[n]))}
    for node in order:
        count = node_counts.get(node, 0)
        size = S_MIN + (S_MAX - S_MIN) * (count / max_count)
        color = cmap(0.25 + 0.65 * (count / max_count) ** 0.5) if count else "#EEEEEE"
        ax_tree.scatter([xpos[node]], [ypos[node]], s=size, color=color,
                         edgecolors="#4C4C4C", linewidths=0.8, zorder=2)
        if node in LEAF_GROUPS:
            up = leaf_rank[node] % 2 == 0
            xytext, va = (0, 14 if up else -20), ("bottom" if up else "top")
        else:
            xytext, va = (-45, 0), "center"
        ax_tree.annotate(f"{node} ({count:,})", (xpos[node], ypos[node]),
                          textcoords="offset points", xytext=xytext,
                          ha="center" if node in LEAF_GROUPS else "right", va=va, fontsize=8.5)

    # --- size legend: reference bubbles at round counts, so absolute magnitude can
    # be read off the plot instead of guessed by comparing circles to each other ---
    legend_counts = sorted({v for v in (50, 500, 5000, max_count) if v <= max_count})
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cmap(0.25 + 0.65 * (v / max_count) ** 0.5),
               markeredgecolor="#4C4C4C", markersize=(S_MIN + (S_MAX - S_MIN) * (v / max_count)) ** 0.5 * 0.9,
               label=f"{v:,}")
        for v in legend_counts
    ]
    ax_tree.legend(handles=legend_handles, title="GO terms originados", loc="lower right",
                   fontsize=8.5, title_fontsize=9, labelspacing=1.6, borderpad=1.2, frameon=False)

    ax_tree.set_xlabel("Distancia evolutiva desde la raiz (nº de nodos)")
    ax_tree.set_yticks([])
    ax_tree.set_ylim(min(ypos.values()) - 1.5, max(ypos.values()) + 1.5)
    ax_tree.set_title(f"Filoestratigrafia de GO terms -- origen evolutivo por nodo\n"
                       f"(n={sum(node_counts.values()):,} GO terms situados, {len(unresolved):,} sin resolver)")
    for spine in ("top", "right", "left"):
        ax_tree.spines[spine].set_visible(False)

    # --- plot 2: total GO terms originated per node, log scale (huge dynamic range:
    # Root/Eukaryota in the thousands, individual lineages in the tens) ---
    nodes_with_counts = [n for n in order if node_counts.get(n, 0) > 0]
    totals = np.array([node_counts[n] for n in nodes_with_counts])
    ax_total.bar(range(len(nodes_with_counts)), totals, width=0.65,
                 color="#4C9BE8", edgecolor="white", linewidth=0.5)
    ax_total.set_yscale("log")
    ax_total.set_xticks(range(len(nodes_with_counts)))
    ax_total.set_xticklabels(nodes_with_counts, rotation=45, ha="right", fontsize=8.5)
    ax_total.set_ylabel("GO terms originados (escala log)")
    ax_total.set_title("Magnitud de innovacion funcional por nodo (recuento bruto)")
    for spine in ("top", "right"):
        ax_total.spines[spine].set_visible(False)

    # --- plot 3: same counts, normalized by the number of species backing each node
    # (its own species if a leaf, or the sum across every descendant leaf if an
    # internal node) -- corrects the sampling-effort bias in plot 2, where Fungi/
    # Metazoa (hundreds of species) mechanically rack up more raw originations than
    # Glaucophyta/pteridophyte (a handful of species) regardless of true innovation ---
    rates = np.array([node_rate[n] for n in nodes_with_counts])
    ax_rate.bar(range(len(nodes_with_counts)), rates, width=0.65,
                color="#F5A623", edgecolor="white", linewidth=0.5)
    ax_rate.set_xticks(range(len(nodes_with_counts)))
    ax_rate.set_xticklabels(nodes_with_counts, rotation=45, ha="right", fontsize=8.5)
    ax_rate.set_ylabel("GO terms originados / especie")
    ax_rate.set_title("Magnitud de innovacion funcional por nodo (normalizada por nº de especies)")
    for spine in ("top", "right"):
        ax_rate.spines[spine].set_visible(False)

    # --- caveat: this only corrects for how many species were SAMPLED, not for how
    # thoroughly each one was ANNOTATED. Checked directly against
    # merged_species_stats_belen.tsv: Metazoa species average GO/Prot_fan ~9.5 vs
    # ~4.5-5.6 for every other group (roughly double), with LOWER mean IC_fan than
    # everyone else (~10.6 vs ~12.2-13.1) -- i.e. more GO terms per protein but on
    # average less specific ones. Consistent with FANTASIA's embedding-similarity
    # transfer favoring Metazoa proteins because its experimentally-characterized
    # reference set (Swiss-Prot) is itself dominated by well-studied model animals
    # -- so Metazoa's outsized rate here may be substantially an annotation-density
    # artifact of the prediction method, not (only) real evolutionary innovation. ---
    if "Metazoa" in nodes_with_counts:
        idx = nodes_with_counts.index("Metazoa")
        ax_rate.annotate(
            "Aviso: Metazoa tiene ~2x GO/proteina que el resto de grupos\n"
            "(sesgo de anotacion FANTASIA hacia especies modelo bien\n"
            "caracterizadas, no necesariamente mas innovacion real)",
            xy=(idx, rates[idx]), xycoords="data",
            xytext=(0.55, 0.88), textcoords="axes fraction",
            fontsize=7.5, color="#555555", ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    # --- plot 3: category composition per node, normalized to 100% -- decouples
    # "how much" (plot 2) from "what kind" (this plot); a linear stack of raw counts
    # would hide composition shifts in every node dwarfed by Root/Eukaryota/Metazoa ---
    cats = ["biological_process", "molecular_function", "cellular_component", "unknown"]
    bottoms = np.zeros(len(nodes_with_counts))
    for cat in cats:
        vals = np.array([node_category_counts[n].get(cat, 0) for n in nodes_with_counts], dtype=float)
        fracs = vals / totals * 100
        if vals.sum() == 0:
            continue
        ax_frac.bar(range(len(nodes_with_counts)), fracs, bottom=bottoms, width=0.65,
                    color=CATEGORY_COLOR[cat], label=cat, edgecolor="white", linewidth=0.5)
        bottoms += fracs

    ax_frac.set_xticks(range(len(nodes_with_counts)))
    ax_frac.set_xticklabels(nodes_with_counts, rotation=45, ha="right", fontsize=8.5)
    ax_frac.set_ylabel("% de GO terms originados")
    ax_frac.set_ylim(0, 100)
    ax_frac.set_title("Composicion por categoria GO de las funciones originadas en cada nodo")
    ax_frac.legend(fontsize=8.5, loc="lower right", ncol=2)
    for spine in ("top", "right"):
        ax_frac.spines[spine].set_visible(False)

    for fmt in plot_formats:
        out_path = Path(args.output).with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")

    # --- textual highlight: top-IC novel functions at a few key nodes ---
    highlight_nodes = [n for n in ["Root", "Eukaryota", "Opisthokonta", "Metazoa", "Fungi",
                                    "Archaeplastida", "angiosperms"] if node_counts.get(n, 0) > 0]
    for node in highlight_nodes:
        members = [go_id for go_id, (n, _) in origin_by_go.items() if n == node]
        members_ic = [(go_id, go_ic.get(go_id, 0.0)) for go_id in members]
        members_ic.sort(key=lambda t: t[1], reverse=True)
        print(f"\n--- Novel functions originating at '{node}' (n={len(members)}, top {args.top_n} by IC) ---")
        for go_id, ic in members_ic[:args.top_n]:
            print(f"    {go_id}  IC={ic:.2f}  {go_desc.get(go_id, '?')}")


if __name__ == "__main__":
    main()
