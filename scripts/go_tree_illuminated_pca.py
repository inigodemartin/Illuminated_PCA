#!/usr/bin/env python3
# hellooooooooo
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from illuminate_PCA import load_taxonomy,load_go_obo,get_descendant_gos,count_descendant_gos,run_illuminated_PCA
from graphviz import Digraph
from collections import defaultdict, deque
import argparse
import os
import subprocess
from PIL import Image

def get_go_relations(go_id, obo_file, part_of=True):


    child_to_parents = defaultdict(set)
    parent_to_children = defaultdict(set)

    go_desc = {}
    obsolete = set()
    current_id = None

    with open(obo_file, "r") as f:

        for line in f:

            line = line.strip()

            if line == "[Term]":
                current_id = None

            elif line.startswith("id:"):

                current_id = line.split("id:")[1].strip()

            elif line.startswith("name:") and current_id is not None:

                go_desc[current_id] = (
                    line.split("name:")[1].strip()
                )

            elif line == "is_obsolete: true" and current_id is not None:

                obsolete.add(current_id)

            elif line.startswith("is_a:") and current_id is not None:

                parent = (
                    line.split("is_a:")[1]
                    .split("!")[0]
                    .strip()
                )

                # Child -> parent
                child_to_parents[current_id].add(parent)

                # Parent -> child
                parent_to_children[parent].add(current_id)

            # =========================
            # NEW: part_of relations
            # =========================
            elif part_of and line.startswith("relationship: part_of") and current_id is not None:
                parent = line.split("part_of")[1].split("!")[0].strip()

                child_to_parents[current_id].add(parent)
                parent_to_children[parent].add(current_id)
    # =========================
    # Ancestors
    # =========================

    ancestors = {}

    queue = deque([(go_id, 0)])
    visited = set()

    while queue:

        current, level = queue.popleft()

        for parent in child_to_parents.get(current, []):

            if parent not in visited and parent not in obsolete:

                visited.add(parent)

                ancestors[parent] = {
                    "desc": go_desc.get(parent, "unknown"),
                    "level": level + 1
                }

                queue.append((parent, level + 1))

    # =========================
    # Descendants
    # =========================

    descendants = {}

    queue = deque([(go_id, 0)])
    visited = set()

    while queue:

        current, level = queue.popleft()

        for child in parent_to_children.get(current, []):

            if child not in visited and child not in obsolete:

                visited.add(child)

                descendants[child] = {
                    "desc": go_desc.get(child, "unknown"),
                    "level": level + 1
                }

                queue.append((child, level + 1))

    return ancestors,descendants,go_desc,child_to_parents,parent_to_children
    


def _nid(go_id):
    """
    Sanitize GO ID for graphviz.
    Colons cause graphviz parsing issues.
    """

    return go_id.replace(":", "_")



def wrap_text(text, width=40):
    """Wrap text every N characters (Graphviz-safe)"""
    words = text.split()
    lines = []
    current = ""

    for w in words:
        if len(current) + len(w) + 1 <= width:
            current += (" " + w if current else w)
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)

    # Cambiado: usar BR ALIGN="LEFT" en lugar de BR solo
    return "<BR ALIGN=\"LEFT\"/>".join(lines)

def sanitize(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------- #
#  Node rendering: every node gets the same fixed size, regardless of    #
#  how long its GO description is.                                       #
# ---------------------------------------------------------------------- #

NODE_WIDTH = 220        # fixed width (pt) for every tree node
IMG_HEIGHT = 160         # fixed height (pt) for the embedded PCA plot
DESC_WRAP_WIDTH = 28     # characters per description line
DESC_MAX_LINES = 2       # description lines before truncating with "…"


def enhance_image_quality(png_path, target_size=(1000, 1000)):
    """
    Improve image quality before inserting into a Graphviz node.
    """
    if not os.path.exists(png_path):
        return png_path

    enhanced_path = png_path.replace('.png', '_enhanced.png')

    try:
        with Image.open(png_path) as img:

            img.thumbnail(target_size, Image.Resampling.LANCZOS)

            from PIL import ImageEnhance

            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)

            img.save(enhanced_path, 'PNG', optimize=False, dpi=(300, 300))
            return enhanced_path

    except Exception as e:
        print(f"Warning: Could not enhance {png_path}: {e}")
        return png_path


def format_description(description):
    """
    Sanitize, wrap and truncate a GO description to at most
    DESC_MAX_LINES lines of DESC_WRAP_WIDTH characters each, so it never
    makes a node wider or noticeably taller than its fixed size.
    """
    br = '<BR ALIGN="LEFT"/>'

    text = sanitize(description)
    lines = wrap_text(text, width=DESC_WRAP_WIDTH).split(br)

    if len(lines) > DESC_MAX_LINES:
        lines = lines[:DESC_MAX_LINES]
        lines[-1] = lines[-1].rstrip() + "…"

    return br.join(lines)


def create_node(dot, go_id, description, png_path, highlight=False):
    """
    Add a fixed-size node with the GO ID, its (wrapped) description and
    its illuminated PCA plot.
    """
    enhanced_png = enhance_image_quality(png_path)
    abs_path = os.path.abspath(enhanced_png)

    header_bg = "#ffffcc" if highlight else "#6FACE8"

    safe_go_id = sanitize(go_id)
    safe_description = format_description(description)

    label = f'''<
<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" WIDTH="{NODE_WIDTH}" ALIGN="LEFT">
    <TR><TD WIDTH="{NODE_WIDTH}" BGCOLOR="{header_bg}" ALIGN="LEFT" VALIGN="TOP">
            <FONT POINT-SIZE="12"><B>{safe_go_id}</B></FONT><BR ALIGN="LEFT"/>
            <FONT POINT-SIZE="10">{safe_description}</FONT>
        </TD></TR>
    <TR><TD WIDTH="{NODE_WIDTH}" HEIGHT="{IMG_HEIGHT}" FIXEDSIZE="TRUE"><IMG SRC="{abs_path}" SCALE="TRUE"/></TD></TR>
</TABLE>>'''

    dot.node(_nid(go_id), label=label)


def add_level_ordering(dot, nodes_by_level):
    """
    Force all nodes at the same tree depth onto the same rank, ordered
    left-to-right by GO ID. Together with sorting node/edge creation
    elsewhere, this makes the rendered tree fully reproducible: the same
    query always produces the same layout.
    """
    for level in sorted(nodes_by_level):
        ids_sorted = sorted(nodes_by_level[level])

        with dot.subgraph() as s:
            s.attr(rank="same")

            for go_id in ids_sorted:
                s.node(_nid(go_id))

            for a, b in zip(ids_sorted, ids_sorted[1:]):
                s.edge(_nid(a), _nid(b), style="invis")


def plot_go_descendants(go_id, descendants, go_desc, parent_to_children, output_file="go_descendants"):

    dot = Digraph(comment=f"GO descendants of {go_id}")

    dot.attr(rankdir="TB", bgcolor="white")
    dot.attr("node", shape="plain")
    dot.attr(nodesep="0.3")
    dot.attr(ranksep="0.4")
    dot.attr(splines="polyline")
    dot.attr(ratio="compress")

    # Query node
    query_png = f"illuminated_pca_{go_id.replace(':', '_')}.png"
    create_node(dot, go_id, go_desc.get(go_id, "unknown"), query_png, highlight=True)

    # Descendant nodes, in a fixed (sorted) order for reproducibility
    for desc_id, info in sorted(descendants.items()):
        desc_png = f"illuminated_pca_{desc_id.replace(':', '_')}.png"
        create_node(dot, desc_id, info["desc"], desc_png, highlight=False)

    # Edges
    all_nodes = set(descendants.keys()) | {go_id}

    for parent in sorted(all_nodes):
        for child in sorted(parent_to_children.get(parent, [])):
            if child in all_nodes:
                dot.edge(_nid(parent), _nid(child))

    # Order nodes within each tree depth by GO ID so the layout is
    # identical across runs
    nodes_by_level = defaultdict(list)
    nodes_by_level[0].append(go_id)
    for desc_id, info in descendants.items():
        nodes_by_level[info["level"]].append(desc_id)

    add_level_ordering(dot, nodes_by_level)

    # Save graphviz source
    gv_file = f"{output_file}.gv"

    dot.save(gv_file)

    # Render PNG
    subprocess.run(
        f'dot -Tpng -Gdpi=300 -Gsize="40,40" "{gv_file}" -o "{output_file}.png"',
        shell=True
    )

    # Remove temporary .gv
    os.remove(gv_file)

    # Remove temporary enhanced images
    for desc_id in descendants:

        desc_enhanced = (
            f"illuminated_pca_{desc_id.replace(':', '_')}_enhanced.png"
        )

        if os.path.exists(desc_enhanced):
            os.remove(desc_enhanced)

    query_enhanced = (
        f"illuminated_pca_{go_id.replace(':', '_')}_enhanced.png"
    )

    if os.path.exists(query_enhanced):
        os.remove(query_enhanced)

    return f"{output_file}.png"

def plot_go_ancestors(go_id, ancestors, go_desc, child_to_parents, output_file="go_ancestors"):

    dot = Digraph(comment=f"GO ancestors of {go_id}")

    dot.attr(rankdir="TB", bgcolor="white")
    dot.attr("node", shape="plain")
    dot.attr(nodesep="0.3")
    dot.attr(ranksep="0.4")
    dot.attr(splines="polyline")
    dot.attr(ratio="compress")

    # Query node
    query_png = f"illuminated_pca_{go_id.replace(':', '_')}.png"
    create_node(dot, go_id, go_desc.get(go_id, "unknown"), query_png, highlight=True)

    # Ancestor nodes, in a fixed (sorted) order for reproducibility
    for anc_id, info in sorted(ancestors.items()):
        anc_png = f"illuminated_pca_{anc_id.replace(':', '_')}.png"
        create_node(dot, anc_id, info["desc"], anc_png, highlight=False)

    # Edges
    all_nodes = set(ancestors.keys()) | {go_id}
    for child in sorted(all_nodes):
        for parent in sorted(child_to_parents.get(child, [])):
            if parent in all_nodes:
                dot.edge(_nid(parent), _nid(child))

    # Order nodes within each tree depth by GO ID so the layout is
    # identical across runs
    nodes_by_level = defaultdict(list)
    nodes_by_level[0].append(go_id)
    for anc_id, info in ancestors.items():
        nodes_by_level[info["level"]].append(anc_id)

    add_level_ordering(dot, nodes_by_level)

    # Save graphviz source and render
    gv_file = f"{output_file}.gv"
    dot.save(gv_file)

    subprocess.run(
        f'dot -Tpng -Gdpi=300 -Gsize="40,40" "{gv_file}" -o "{output_file}.png"',
        shell=True
    )

    os.remove(gv_file)

    for anc_id in ancestors:
        anc_enhanced = f"illuminated_pca_{anc_id.replace(':', '_')}_enhanced.png"
        if os.path.exists(anc_enhanced):
            os.remove(anc_enhanced)

    query_enhanced = f"illuminated_pca_{go_id.replace(':', '_')}_enhanced.png"
    if os.path.exists(query_enhanced):
        os.remove(query_enhanced)

    return f"{output_file}.png"

def parse_args():
    parser = argparse.ArgumentParser(description="GO ancestor graph with PCA images")

    parser.add_argument("--go","-g",type=str,required=True,help="GO ID")
    parser.add_argument("--matrix", "-m",type=str,required=True,help="Input matrix")
    parser.add_argument("--taxa","-t",type=str,default=None,help="Taxa filter")
    parser.add_argument("--update",action="store_true",help="Update cached files")
    parser.add_argument("-o", "--no_outliers",action="store_true",help="Take outliers into account in PCA scaling")
    parser.add_argument("-d", "--count_descendants", action="store_true", default=None, help="Include GO terms that are descendants of the query GO term.") 
    parser.add_argument("-p", "--plot_descendants", action="store_true", default=None, help="Plot descendant GOs tree of the query.") 

    return parser.parse_args()




def run_pca_job(args_tuple):
    """
    Wrapper for run_illuminated_PCA that accepts a single tuple argument
    (required for multiprocessing pickling).
    """
    matrix, go_counts, taxon_dict, target_go, go_desc_local, taxa, no_outliers = args_tuple
    run_illuminated_PCA(matrix, go_counts, taxon_dict, target_go, go_desc_local, taxa, no_outliers)
    return target_go


def collect_pca_jobs(
    go_terms,
    matrix,
    descendants,
    taxon_dict,
    go_descriptions,
    go_desc,
    taxa,
    no_outliers,
    count_descendant,
):
    """
    Build a list of (args_tuple) for every PCA job that still needs to run.
    Skips jobs whose output PNG already exists and is non-empty.
    """
    jobs = []
    for target_go in go_terms:
        out_png = Path(f"illuminated_pca_{target_go.replace(':', '_')}.png")
        if out_png.exists() and out_png.stat().st_size > 0:
            print(f"[skip] {out_png} already exists.")
            continue

        if count_descendant:
            go_counts = count_descendant_gos(matrix, descendants)
            desc_arg = go_descriptions
        else:
            go_counts = count_descendant_gos(matrix, [target_go])
            desc_arg = go_desc

        jobs.append((matrix, go_counts, taxon_dict, target_go, desc_arg, taxa, no_outliers))
    return jobs


def run_jobs_parallel(jobs, max_workers=None):
    """
    Execute PCA jobs in parallel.
    max_workers defaults to os.cpu_count() if not specified.
    """
    if not jobs:
        print("No PCA jobs to run.")
        return

    workers = max_workers or os.cpu_count()
    print(f"Running {len(jobs)} PCA job(s) across {workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_pca_job, job): job[3] for job in jobs}  # job[3] = target_go
        for future in as_completed(futures):
            target_go = futures[future]
            try:
                result = future.result()
                print(f"[done] {result}")
            except Exception as exc:
                print(f"[error] {target_go} raised: {exc}")


def main():
    args = parse_args()

    go             = args.go
    taxa           = args.taxa
    no_outliers    = args.no_outliers
    update         = args.update
    matrix         = args.matrix
    count_descendant  = args.count_descendants
    plot_descendants  = args.plot_descendants

    taxon_file = "/data/users/demartini/FANTASIA_project/plots_2025/merged_taxons.tsv"
    obo_file   = "/data/users/demartini/DB/go-basic_2025.obo"

    taxon_dict = load_taxonomy(taxon_file)

    ancestors, descendants, go_desc, child_to_parents, parent_to_children = get_go_relations(go, obo_file)

    children_map, go_descriptions = load_go_obo(obo_file)

    # ------------------------------------------------------------------ #
    #  Build the full list of GO terms to process, then run in parallel   #
    # ------------------------------------------------------------------ #
    if plot_descendants:
        go_terms = [go] + list(descendants)
    else:
        go_terms = [go] + list(ancestors)

    jobs = collect_pca_jobs(
        go_terms=go_terms,
        matrix=matrix,
        descendants=descendants,
        taxon_dict=taxon_dict,
        go_descriptions=go_descriptions,
        go_desc=go_desc,
        taxa=taxa,
        no_outliers=no_outliers,
        count_descendant=count_descendant,
    )

    run_jobs_parallel(jobs, max_workers=16)          # <-- all PCA calls happen here, in parallel

    # ------------------------------------------------------------------ #
    #  Graph rendering (quick, kept sequential)                           #
    # ------------------------------------------------------------------ #
    if plot_descendants:
        plot_go_descendants(
            go, descendants, go_desc, parent_to_children,
            output_file=f"{go.replace(':', '_')}_descendants",
        )
    else:
        plot_go_ancestors(
            go, ancestors, go_desc, child_to_parents,
            output_file=f"{go.replace(':', '_')}_ancestors",
        )


if __name__ == "__main__":
    main()