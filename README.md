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
| `scripts/interactive_go_tree.py` | Generates a single self-contained interactive HTML page for a GO ancestor/descendant tree: click a node to expand its illuminated PCA inline (zoomable/pannable), hover a point for species details. |

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

`scripts/interactive_go_tree.py` only needs `numpy`, `pandas` and
`scikit-learn` — no `graphviz`/`Pillow`, since it emits HTML/JSON instead of
PNGs.

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

## Interactive HTML explorer

`scripts/interactive_go_tree.py` generates a single self-contained HTML
file for the same kind of GO ancestor/descendant tree as
`go_tree_illuminated_pca.py`, but interactive instead of a static PNG:
click a node to expand its illuminated PCA inline (scroll to zoom, drag to
pan, "Reset zoom" to go back), hover a point to see the species, its GO
count, its total protein count, and what share of its proteome that GO
term represents. No server or internet connection is needed — the file
works fully offline, just open it in a browser.

A PCA always expands right under its own node, in that level's row — not
in some unrelated part of the page — so the tree stays easy to read even
with several open at once: panels opened from the same level share that
row side by side, wrapping/shrinking to fit, while levels below are
simply pushed down, same as any inline accordion. "Expand all PCAs"/
"Collapse all PCAs" buttons in the header open or close every node at
once. A fixed sidebar holds the taxon legend/select-all-none (one shared
copy that controls every open panel, instead of repeating it in each
one), a "Min"/"Max" count threshold per currently-open PCA: set either
to only plot species whose GO count for that node falls in that range,
e.g. min=10 hides every species annotated fewer than 10 times for that
term, and a "Top GO terms per PC" list — the GO terms whose relative
abundance most drives PC1/PC2 (the fitted PCA's own loadings, sign and
all), a property of the PCA itself rather than of any one node, so it's
shown once rather than per panel. Each panel can be downloaded
individually as a PNG (title, axis variance labels and taxon legend
included), and "Download tree PNG" exports the whole tree exactly as
shown on screen — boxes, connectors and every currently-expanded PCA in
place, thresholds and all — as one image. Each node also shows its GO
Information Content (IC), from the bundled `data/All_GOs_ic.tsv`.

Differences from the PNG pipeline:

- Takes a single **raw GO counts matrix** (no separate pre-normalized
  matrix file). Relative abundance (`count / Total_prots`) is computed
  internally before running the PCA.
- Needs one extra input: a **species-stats TSV** with a `Total_prots`
  column (total protein count per species) — used both for the internal
  normalization and to show "share of proteome" in the tooltip.
- The PCA is computed once and reused for every node, instead of being
  recomputed per node like in the PNG pipeline (the layout never actually
  depends on which GO term is illuminated).
- Taxonomy and OBO file paths are explicit CLI flags, not hardcoded.
- Taxonomic-group filtering is also available live in the browser
  (checkboxes), on top of whatever `-t` filter was baked in at generation
  time.

### Usage

```bash
python scripts/interactive_go_tree.py \
  -g GO:0008152 \
  -m raw_counts_matrix.tsv \
  --species-stats species_stats.tsv \
  --taxonomy taxonomy.tsv \
  --obo go-basic.obo
```

| Flag | Description |
|---|---|
| `-g, --go` | Root GO ID (required). |
| `-m, --matrix` | Raw GO counts matrix, species x GO terms (required). |
| `--species-stats` | TSV with a `Species` index and a `Total_prots` column (required). |
| `--taxonomy` | TSV with `Species` and `Group` columns (required). |
| `--obo` | GO OBO file (required). |
| `--ic-file` | GO Information Content TSV (default: bundled `data/All_GOs_ic.tsv`). |
| `-t, --taxa` | Restrict to these taxonomic groups. |
| `-d, --count_descendants` | Sum counts over each node's own descendants too. |
| `-o, --no_outliers` | Percentile-clipped scaling instead of log scaling. |
| `-p, --plot_descendants` | Build the tree from descendants instead of ancestors. |
| `--output` | Output HTML path (default: `interactive_<GO_ID>_<ancestors\|descendants>.html`). |
| `--top-loadings-n` | Most-influential GO terms to report per PC (default: 20). |
| `--loadings-output` | Top-loadings TSV path (default: alongside `--output`, with `_top_loadings.tsv`). |

### Output

A single HTML file with the GO tree laid out level by level; clicking a
node expands an inline panel with its illuminated PCA. All data (PCA
coordinates, GO counts, tree structure, top loadings) is embedded as JSON
in the file itself, so it has no external dependencies and works fully
offline.

A second file, `<output>_top_loadings.tsv`, lists the same top GO terms
per PC shown in the sidebar — one row per (PC, rank, GO id, description,
signed loading) — for use outside the browser (spreadsheets, downstream
scripts, etc.).
