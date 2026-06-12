# Illuminated PCA

Tools for visualizing **Gene Ontology (GO) annotation profiles** across species
using PCA, with an "illumination" mode that highlights how strongly each
species/taxon is annotated for a specific GO term (or its full ontology
sub-tree).

Built for the **FANTASIA project**, where each species has a matrix of GO
term counts derived from functional annotation.

## Contents

| Script | Purpose |
|---|---|
| `scripts/illuminate_PCA.py` | Core module: runs PCA on a GO-count matrix and plots it, either as a standard taxonomy-colored PCA or as an "illuminated" PCA for one or more GO terms. |
| `scripts/go_tree_illuminated_pca.py` | Builds on the core module to generate illuminated PCA plots for an entire GO ancestor/descendant tree, then renders that tree as a Graphviz diagram with each PCA plot embedded in its node. |

## How it works

1. **PCA on the GO matrix** — Species are rows, GO terms are columns
   (counts). Rare GO terms are dropped, the data is standardized
   (`StandardScaler`), and reduced to 2 dimensions with `TruncatedSVD`
   (suitable for large/sparse matrices). Extreme points are removed via
   percentile filtering.

2. **Standard plot** — Points are colored by taxonomic group (from a
   species → group mapping file), with a stable color palette shared across
   runs.

3. **Illuminated plot** — For a chosen GO term, each species' total count for
   that term (optionally including all of its descendants in the ontology)
   is computed. Points are then drawn with **size and opacity proportional to
   that count** (log-scaled, with optional outlier-robust scaling), so
   heavily-annotated species "light up" while poorly-annotated ones fade out.
   The plot is saved as `illuminated_pca_<GO_ID>.png`.

4. **GO tree view** — Given a GO term, its ancestors (or descendants, with
   `-p`) are extracted from a GO OBO file (including `is_a` and `part_of`
   relations, excluding obsolete terms). An illuminated PCA is generated for
   every term in that tree **in parallel**, then a Graphviz diagram is
   rendered showing the GO hierarchy with each node's PCA plot embedded
   inside it.

## Requirements

- Python 3
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- `Pillow` (PIL)
- [`graphviz`](https://pypi.org/project/graphviz/) Python package **and** the
  Graphviz `dot` binary available on `PATH` (only needed for
  `go_tree_illuminated_pca.py`)

```bash
pip install numpy pandas scikit-learn matplotlib pillow graphviz
# Graphviz binary (Debian/Ubuntu):
sudo apt install graphviz
```

## Input files

- **GO matrix (`-m/--matrix`)** — TSV, species as row index, GO IDs as
  columns, raw annotation counts as values.
- **Taxonomy file** — TSV with `Species` and `Group` columns, mapping each
  species/sample to a taxonomic group for coloring.
- **GO OBO file** — a standard `go-basic.obo` ontology file, used to resolve
  GO term descriptions and parent/child relationships.

> ⚠️ The taxonomy file and OBO file paths are currently hardcoded at the top
> of `main()` in both scripts
> (`/data/users/demartini/FANTASIA_project/plots_2025/merged_taxons.tsv` and
> `/data/users/demartini/DB/go-basic_2025.obo`). Update these paths to match
> your environment before running.

## Usage

### `illuminate_PCA.py`

```bash
python scripts/illuminate_PCA.py -m matrix.tsv [options]
```

| Flag | Description |
|---|---|
| `-m, --matrix` | Path to the GO-count matrix (TSV). |
| `-g, --go` | Comma-separated GO IDs to illuminate. If omitted, a standard taxonomy-colored PCA is plotted instead. |
| `-t, --taxa` | One or more taxonomic groups to include (default: all). |
| `-d, --count_descendants` | Include counts from all descendants of each given GO term, not just the term itself. |
| `-o, --no_outliers` | Apply robust scaling so extreme abundance values don't dominate the opacity/size scale. |

**Examples**

```bash
# Standard PCA, colored by taxonomic group
python scripts/illuminate_PCA.py -m matrix.tsv

# Illuminated PCA for a single GO term
python scripts/illuminate_PCA.py -m matrix.tsv -g GO:0008152

# Illuminate using a GO term plus all of its descendants,
# restricted to two taxonomic groups
python scripts/illuminate_PCA.py -m matrix.tsv -g GO:0008152 -d -t Algae Fungi
```

### `go_tree_illuminated_pca.py`

```bash
python scripts/go_tree_illuminated_pca.py -g GO:0008152 -m matrix.tsv [options]
```

| Flag | Description |
|---|---|
| `-g, --go` | Reference GO ID (required). |
| `-m, --matrix` | Path to the GO-count matrix (required). |
| `-t, --taxa` | Taxonomic group(s) to include. |
| `-d, --count_descendants` | Sum counts over each tree node's descendants rather than the node itself. |
| `-o, --no_outliers` | Robust scaling for opacity/size (same as above). |
| `-p, --plot_descendants` | Build the tree from the term's **descendants** instead of its ancestors. |
| `--update` | Update cached files (currently unused). |

**Examples**

```bash
# Ancestor tree, one illuminated PCA per ancestor node
python scripts/go_tree_illuminated_pca.py -g GO:0008152 -m matrix.tsv

# Descendant tree instead of ancestors
python scripts/go_tree_illuminated_pca.py -g GO:0008152 -m matrix.tsv -p

# Descendant tree, each node aggregating its own descendant counts
python scripts/go_tree_illuminated_pca.py -g GO:0008152 -m matrix.tsv -p -d
```

## Output

- `illuminated_pca_<GO_ID>.png` — one illuminated PCA plot per GO term
  processed (skipped on re-run if it already exists and is non-empty).
- `<GO_ID>_ancestors.png` / `<GO_ID>_descendants.png` — the rendered Graphviz
  tree with each node's PCA plot embedded.

## Notes

- PCA jobs for an entire GO tree run in parallel via
  `ProcessPoolExecutor` (defaults to 16 workers).
- Color assignment per taxonomic group is deterministic and stable across
  runs/subsets, so the same group always gets the same color.
