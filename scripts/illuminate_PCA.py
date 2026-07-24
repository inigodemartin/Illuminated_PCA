#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

from collections import defaultdict, deque
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def load_taxonomy(taxon_file):
    """
    Load species → group mapping.
    """
    tax = pd.read_csv(taxon_file, sep="\t")
    return dict(zip(tax["Species"], tax["Group"]))

def assign_taxonomy_group(df, taxon_dict):
    """
    Map df's index (species) to a "Group" column via taxon_dict and drop rows
    with no match, warning how many were dropped -- species present in the
    matrix/PCA but missing from the taxonomy file used to silently disappear
    from the plot with no indication why.
    """
    df = df.copy()
    df["Group"] = df.index.map(taxon_dict)
    missing = df["Group"].isna()
    if missing.any():
        print(f"Warning: {int(missing.sum())} species have no Group in the taxonomy file — excluded from output")
    return df.dropna(subset=["Group"])

def load_go_obo(obo_file):
    """
    Parse GO OBO file and build:
    1. parent -> children mapping
    2. GO -> description mapping
    """

    from collections import defaultdict

    children = defaultdict(set)
    go_desc = {}

    current_id = None

    with open(obo_file, "r") as f:
        for line in f:

            line = line.strip()

            if line == "[Term]":
                current_id = None

            elif line.startswith("id:"):
                current_id = line.split("id:")[1].strip()

            elif line.startswith("name:") and current_id is not None:
                desc = line.split("name:")[1].strip()
                go_desc[current_id] = desc

            elif line.startswith("is_a:") and current_id is not None:
                parent = line.split("is_a:")[1].split("!")[0].strip()
                children[parent].add(current_id)

    return children, go_desc

def get_descendant_gos(go_id, children_map):
    """
    Return all descendants of a GO term.
    """
    descendants = set()
    queue = deque([go_id])

    while queue:
        current = queue.popleft()

        for child in children_map.get(current, []):
            if child not in descendants:
                descendants.add(child)
                queue.append(child)
    descendants.add(go_id)

    return list(descendants)

def run_pca(go_matrix_tsv):
    """
    PCA using TruncatedSVD for large GO matrices.
    """

    df = pd.read_csv(go_matrix_tsv, sep="\t", index_col=0).fillna(0)

    # Remove very rare GO terms
    df = df.loc[:, df.sum(axis=0) > 5]

    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df.values)

    model = TruncatedSVD(n_components=2)
    components = model.fit_transform(data_normalized)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"],
        index=df.index
    )

    explained_variance = model.explained_variance_ratio_

    return pca_df, explained_variance

def remove_outliers(pca_df, low=1, high=99):
    """
    Remove extreme PCA points using percentile filtering.
    """

    mask = (
        (pca_df["PC1"] >= np.percentile(pca_df["PC1"], low)) &
        (pca_df["PC1"] <= np.percentile(pca_df["PC1"], high)) &
        (pca_df["PC2"] >= np.percentile(pca_df["PC2"], low)) &
        (pca_df["PC2"] <= np.percentile(pca_df["PC2"], high))
    )

    return pca_df[mask]

def generate_distinct_colors(n):
    """
    Generate n deterministic visually distinct colors using HSV.
    The output is stable for a fixed ordering.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.65 + (i % 3) * 0.1
        value = 0.9
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def build_global_color_map(taxon_dict):
    """
    Build a stable color mapping for ALL groups in dataset.
    This ensures consistent colors across different runs/subsets.
    """

    all_groups = sorted(set(taxon_dict.values()))
    n_groups = len(all_groups)

    palette = generate_distinct_colors(n_groups)

    return dict(zip(all_groups, palette))


def plot_pca(pca_df, taxon_dict, explained_variance, selected_taxa=None):
    """
    Plot PCA colored by taxonomic group.
    """

    # Add taxonomy, dropping species with no match
    pca_df = assign_taxonomy_group(pca_df, taxon_dict)

    # Build stable global color map
    color_map = build_global_color_map(taxon_dict)

    # Filter selected taxa if provided
    if selected_taxa:
        pca_df = pca_df[pca_df["Group"].isin(selected_taxa)]

    groups = sorted(pca_df["Group"].unique())
    n_groups = len(groups)

    plt.figure(figsize=(8, 6), dpi=150)

    for group in groups:
        sub = pca_df[pca_df["Group"] == group]

        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            label=group,
            color=color_map[group],
            s=40,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}% variance)")
    plt.title("Fantasia GO-based PCA by taxonomic group")

    plt.axhline(0, linewidth=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, linewidth=0.5, color='gray', linestyle='--', alpha=0.5)

    # Legend formatting
    if n_groups > 15:
        plt.legend(frameon=True, fontsize=6, loc='best', ncol=2)
    else:
        plt.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plt.show()

def run_illuminated_PCA(input_matrix, go_counts, taxon_dict,go, go_desc, taxa=None, no_outliers=None, outlier_percentile=(0, 100)):
    """
    PCA where point opacity and size are controlled by GO abundance.

    Species with higher GO counts appear:
    - more opaque
    - larger

    Species with low counts appear:
    - more transparent
    - smaller

    Counts are log-scaled to improve contrast.
    """
    out_png = Path(f"illuminated_pca_{go.replace(':','_')}.png")

    if out_png.exists() and out_png.stat().st_size > 0:
        print(f"{out_png} already exists.")
        
    else:

                # Run PCA
        pca_df, explained_variance = run_pca(input_matrix)

        # Remove outliers
        pca_df = remove_outliers(pca_df, low=outlier_percentile[0], high=outlier_percentile[1])

        # Add taxonomy, dropping species with no match
        pca_df = assign_taxonomy_group(pca_df, taxon_dict)

        # Filter taxa if requested
        if taxa:
            pca_df = pca_df[pca_df["Group"].isin(taxa)]

        # Add GO counts
        pca_df["GO_count"] = pca_df.index.map(go_counts).fillna(0)

        # Stable colors
        color_map = build_global_color_map(taxon_dict)

        # Maximum count
        max_count = pca_df["GO_count"].max()

        # Opacity range
        min_alpha = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000001
        max_alpha = 1.0

        plt.figure(figsize=(8, 6), dpi=150)

        groups = sorted(pca_df["Group"].unique())

        for group in groups:

            sub = pca_df[pca_df["Group"] == group]
            if sub.empty:
                continue

            # Log scaling improves visualization
            if max_count > 0:

                if no_outliers:

                    all_counts = pca_df["GO_count"].astype(float)

                    max_count = all_counts.max()

                    if max_count <= 0:
                        alphas = np.zeros(len(sub))
                    else:
                        # Scale relative to global maximum
                        norm = sub["GO_count"] / max_count

                        # Soft clipping of extreme values (99th percentile)
                        p99 = np.percentile(norm[norm > 0], 99) if np.any(norm > 0) else 1.0

                        norm = np.minimum(norm, p99)

                        if p99 > 0:
                            norm = norm / p99

                        # Gamma correction to enhance low signals
                        norm = norm ** 0.5

                        alphas = min_alpha + norm * (max_alpha - min_alpha)

                        # Species with zero abundance are invisible
                        alphas[sub["GO_count"] == 0] = 0.0
                else:
                    
                    

                    log_counts = np.log1p(sub["GO_count"])
                    log_max = np.log1p(max_count)
                    alphas = (
                        min_alpha +
                        (log_counts / log_max) * (max_alpha - min_alpha)
                    )



            else:
                alphas = np.repeat(min_alpha, len(sub))

            # Plot each point independently
            for (_, row), alpha in zip(sub.iterrows(), alphas):

                # Size linked to opacity
                size = 10 + (alpha * 60)

                if np.isnan(alpha):
                    continue
                plt.scatter(
                    row["PC1"],
                    row["PC2"],
                    color=color_map[group],
                    alpha=alpha,
                    s=size,
                    edgecolors="black",
                    linewidth=0.4
                )

        # Dummy points for legend
        for group in groups:
            plt.scatter(
                [],
                [],
                color=color_map[group],
                label=group,
                s=40
            )

        plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}% variance)")
        plt.suptitle(f"Illuminated PCA for {go}: {go_desc.get(go, go)}", fontsize=14)

        # Reference lines
        plt.axhline(
            0,
            linewidth=0.5,
            color='gray',
            linestyle='--',
            alpha=0.5
        )

        plt.axvline(
            0,
            linewidth=0.5,
            color='gray',
            linestyle='--',
            alpha=0.5
        )

        # Legend
        if len(groups) > 15:
            plt.legend(
                frameon=True,
                fontsize=6,
                loc='best',
                ncol=2
            )
        else:
            plt.legend(
                frameon=True,
                fontsize=8
            )

        plt.tight_layout()
        plt.savefig(
        f"illuminated_pca_{go.replace(':','_')}.png",
        dpi=300,
        bbox_inches='tight'
        )

def run_normal_PCA(input_matrix, taxon_dict, taxa, outlier_percentile=(0, 100)):
    pca_df, explained_variance = run_pca(input_matrix)
    pca_df = remove_outliers(pca_df, low=outlier_percentile[0], high=outlier_percentile[1])
    plot_pca(pca_df, taxon_dict, explained_variance, selected_taxa=taxa)



def count_descendant_gos(matrix, go_list, output_file=None, verbose=False):
    """
    Sum GO counts per species for a given list of GO terms.

    Optionally writes the filtered GO matrix (raw counts) to file.
    """

    import pandas as pd

    df = pd.read_csv(matrix, sep="\t", index_col=0).fillna(0)

    # keep only GO columns that exist in matrix
    valid_gos = [go for go in go_list if go in df.columns]

    # filtered raw matrix
    filtered_df = df[valid_gos]

    # print preview
    if verbose:
        print("\nFiltered GO matrix (raw counts):")
        print(filtered_df)

    # save to file if requested
    if output_file is not None:
        filtered_df.to_csv(output_file, sep="\t")

    # return summed counts (for PCA or downstream use)
    result = filtered_df.sum(axis=1).to_dict()

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Illuminated PCA")
    parser.add_argument("--update","-u", help="Path to GO_dict if matrix needs to be updated",default=None)
    parser.add_argument("--matrix","-m", help="Path to matrix")
    parser.add_argument("--go", "-g", type=lambda s: [item.strip() for item in s.split(",")], help="Comma-separated GO IDs for illuminating PCA", default=None)    
    parser.add_argument("-t", "--taxa", nargs="*", default=None, help="Taxonomic groups to plot. If not provided, all taxa are used.")
    parser.add_argument("-o", "--no_outliers", action="store_true", default=None, help="Apply robust scaling to reduce the visual dominance of extreme outliers in the PCA representation.")
    parser.add_argument("-d", "--count_descendants", action="store_true", default=None, help="Include GO terms that are descendants of the query GO term.")
    parser.add_argument("--outlier-percentile", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                         help="Drop species whose PC1 or PC2 falls outside this percentile range "
                              "(default: 0 100, i.e. no trimming). Pass e.g. '5 95' to trim.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    taxa = args.taxa

    no_outliers = args.no_outliers
    update = args.update
    matrix = args.matrix
    gos = args.go
    count_descendant = args.count_descendants

    taxon_file = "/data/users/demartini/FANTASIA_project/plots_2025/merged_taxons.tsv"
   
    taxon_dict = load_taxonomy(taxon_file)

    # if update is not None:
    #     update_matrix(update)
    
    if gos is not None:
        for go in gos:
            children_map, go_desc = load_go_obo("/data/users/demartini/DB/go-basic_2025.obo")
            if count_descendant:
                desc = get_descendant_gos(go, children_map)
                go_counts = count_descendant_gos(matrix, desc)
                run_illuminated_PCA(matrix, go_counts, taxon_dict,go, go_desc, taxa, no_outliers, args.outlier_percentile)
            else:
                go_counts = count_descendant_gos(matrix, [go])
                run_illuminated_PCA(matrix, go_counts, taxon_dict,go, go_desc, taxa, no_outliers, args.outlier_percentile)


    else:
        run_normal_PCA(matrix, taxon_dict, taxa, args.outlier_percentile)




if __name__ == "__main__":
    main()
