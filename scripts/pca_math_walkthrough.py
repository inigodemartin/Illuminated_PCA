#!/usr/bin/env python3
"""
Renders a PDF that explains, step by step, the math of the abundance PCA
pipeline (interactive_go_tree.run_pca_on_relative_abundance): what each
transformation does, its formula, and a worked example using one real
species and two real GO columns from the actual matrix (one with a raw
count of 0, one with a "typical" small nonzero count) -- so every number
in the PDF is real data, not illustrative filler. Also reports how the
whole dataset's summary statistics (min/max/mean/median/sd) evolve at
each stage.

Picks the example species deterministically (closest Total_prots to the
dataset median -- a "typical" species, not an outlier) and the example GO
columns from that species' own row (closest-to-median nonzero count, and
the first zero count), so re-running against the same matrix always
reproduces the same walkthrough.
"""

from pathlib import Path
import argparse
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from interactive_go_tree import load_species_stats
from general_pca_common import DEFAULT_IC_PATH, load_go_descriptions

matplotlib.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
})

BLUE = "#4C9BE8"
GREY = "#888888"
AMBER = "#F5A623"
PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", "-m", default="merged_PCA_fantasia.tsv", help="Raw GO counts matrix (species x GO, TSV)")
    ap.add_argument("--species-stats", default="merged_species_stats.tsv", help="TSV with a Species index and Total_prots column")
    ap.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    ap.add_argument("--output", default="pca_math_walkthrough.pdf", help="Output PDF path")
    return ap.parse_args()


def summary_stats(arr):
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "sd": float(flat.std()),
    }


def new_page(pdf, figsize=PAGE_SIZE):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def title_block(ax, y, title, subtitle=None):
    ax.text(0.07, y, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.07, y - 0.045, subtitle, fontsize=11, color=GREY, va="top")


def formula_block(ax, y, formula, fontsize=20):
    ax.text(0.5, y, formula, fontsize=fontsize, ha="center", va="top")


def body_text(ax, y, text, fontsize=11.5, linespacing=1.6):
    ax.text(0.07, y, text, fontsize=fontsize, va="top", ha="left", wrap=True, linespacing=linespacing)


def example_table(ax, y, headers, rows, col_x, fontsize=10.5):
    """Small hand-drawn monospace table: headers + rows of (label, val_zero, val_nonzero)."""
    ax.text(col_x[0], y, headers[0], fontsize=fontsize, fontweight="bold", va="top")
    ax.text(col_x[1], y, headers[1], fontsize=fontsize, fontweight="bold", va="top", color=GREY)
    ax.text(col_x[2], y, headers[2], fontsize=fontsize, fontweight="bold", va="top", color=BLUE)
    y -= 0.035
    ax.plot([col_x[0], 0.93], [y + 0.012, y + 0.012], color=GREY, linewidth=0.8, transform=ax.transAxes)
    for label, v0, v1 in rows:
        ax.text(col_x[0], y, label, fontsize=fontsize, va="top", family="monospace")
        ax.text(col_x[1], y, v0, fontsize=fontsize, va="top", family="monospace", color=GREY)
        ax.text(col_x[2], y, v1, fontsize=fontsize, va="top", family="monospace", color=BLUE)
        y -= 0.032
    return y


def main():
    args = parse_args()

    print(f"Loading {args.matrix} ...")
    header_cols = pd.read_csv(args.matrix, sep="\t", nrows=0).columns
    dtype_map = {c: "float32" for c in header_cols[1:]}
    raw_df = pd.read_csv(args.matrix, sep="\t", index_col=0, dtype=dtype_map).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    go_desc = load_go_descriptions(args.ic_file)

    species = [s for s in raw_df.index if s in total_prots.index]
    raw_df = raw_df.loc[species]
    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    go_columns = list(pca_input.columns)
    print(f"Matrix: {raw_df.shape[0]} species x {raw_df.shape[1]} GO columns "
          f"-> {pca_input.shape[1]} kept after rare-column filter (colsum > 5)")
    del raw_df
    gc.collect()

    # --- pick a real, representative (species, GO) example ---
    species_total = total_prots.loc[species]
    target_species = (species_total - species_total.median()).abs().idxmin()
    sp_idx = species.index(target_species)
    sp_total_prots = float(species_total.loc[target_species])
    print(f"Example species: {target_species} (Total_prots={sp_total_prots:.0f}, "
          f"dataset median={species_total.median():.0f})")

    raw_counts = pca_input.to_numpy(dtype="float32")
    del pca_input
    gc.collect()

    row = raw_counts[sp_idx, :]
    zero_positions = np.where(row == 0)[0]
    nonzero = row[row > 0]
    median_nonzero = np.median(nonzero)
    go_idx_nonzero = int(np.argmin(np.abs(np.where(row > 0, row, np.inf) - median_nonzero)))
    go_idx_zero = int(zero_positions[0])
    go_id_zero = go_columns[go_idx_zero]
    go_id_nonzero = go_columns[go_idx_nonzero]
    desc_zero = go_desc.get(go_id_zero, "(sin descripcion)")
    desc_nonzero = go_desc.get(go_id_nonzero, "(sin descripcion)")
    print(f"Example GO (count=0): {go_id_zero} -- {desc_zero}")
    print(f"Example GO (count~median nonzero): {go_id_nonzero} -- {desc_nonzero} "
          f"(count={row[go_idx_nonzero]:.0f}, row median nonzero={median_nonzero:.1f})")

    # --- walk the real pipeline, stage by stage, capturing global stats + the two example cells ---
    stages = {}  # name -> (summary_stats_dict, example_zero_value, example_nonzero_value)

    stages["0. Conteo crudo"] = (
        summary_stats(raw_counts), float(raw_counts[sp_idx, go_idx_zero]), float(raw_counts[sp_idx, go_idx_nonzero])
    )

    pseudo = raw_counts + 1.0
    del raw_counts
    gc.collect()
    stages["1. + pseudo-count (+1)"] = (
        summary_stats(pseudo), float(pseudo[sp_idx, go_idx_zero]), float(pseudo[sp_idx, go_idx_nonzero])
    )

    np.log(pseudo, out=pseudo)  # now holds log(count+1)
    stages["2. log(count+1)"] = (
        summary_stats(pseudo), float(pseudo[sp_idx, go_idx_zero]), float(pseudo[sp_idx, go_idx_nonzero])
    )

    row_means = pseudo.mean(axis=1, keepdims=True)
    row_mean_species = float(row_means[sp_idx, 0])
    pseudo -= row_means  # now holds CLR values
    stages["3. CLR (centrado por fila)"] = (
        summary_stats(pseudo), float(pseudo[sp_idx, go_idx_zero]), float(pseudo[sp_idx, go_idx_nonzero])
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pseudo)
    col_mean_zero, col_std_zero = float(scaler.mean_[go_idx_zero]), float(scaler.scale_[go_idx_zero])
    col_mean_nonzero, col_std_nonzero = float(scaler.mean_[go_idx_nonzero]), float(scaler.scale_[go_idx_nonzero])
    stages["4. StandardScaler (z-score)"] = (
        summary_stats(scaled), float(scaled[sp_idx, go_idx_zero]), float(scaled[sp_idx, go_idx_nonzero])
    )
    del pseudo
    gc.collect()

    print("Fitting final TruncatedSVD (2 components) to report this species' PC1/PC2 ...")
    model = TruncatedSVD(n_components=2)
    pc = model.fit_transform(scaled)
    explained = model.explained_variance_ratio_ * 100
    species_pc1, species_pc2 = float(pc[sp_idx, 0]), float(pc[sp_idx, 1])
    del scaled
    gc.collect()

    # =========================== render the PDF ===========================
    out_path = Path(args.output)
    with PdfPages(out_path) as pdf:

        # --- Page 1: overview ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "De conteos crudos a PCA",
                    "Explicación matemática paso a paso, con un ejemplo real")
        body_text(ax, 0.82,
            "Este documento recorre, paso a paso, la transformación matemática que sufre la matriz de\n"
            "conteos de GO terms antes de entrar en la PCA (interactive_go_tree.run_pca_on_relative_\n"
            "abundance). Cada paso se explica con su fórmula y se ilustra con un ejemplo real extraído\n"
            "directamente de la matriz de datos -- no son números inventados.")
        body_text(ax, 0.66,
            f"Especie de ejemplo (elegida por tener el Total_prots más cercano a la mediana del\n"
            f"dataset, para que sea representativa y no un caso extremo):\n\n"
            f"  {target_species}   (Total_prots = {sp_total_prots:.0f})\n\n"
            f"Dos columnas GO de esa misma especie, elegidas para ilustrar los dos casos posibles:",
            fontsize=11.5)
        body_text(ax, 0.44,
            f"  · GO con conteo = 0  ->  {go_id_zero}\n"
            f"    \"{desc_zero[:78]}\"\n\n"
            f"  · GO con conteo típico (cercano a la mediana de valores no-cero de esta especie)\n"
            f"    ->  {go_id_nonzero}   (conteo real = {row[go_idx_nonzero]:.0f})\n"
            f"    \"{desc_nonzero[:78]}\"",
            fontsize=11)
        body_text(ax, 0.22,
            "Pasos cubiertos:\n"
            "  1. Pseudo-count (+1)   2. log(count+1)   3. CLR (centrado por fila)\n"
            "  4. StandardScaler (z-score por columna)   5. TruncatedSVD (proyección final)\n\n"
            f"Matriz real usada: {len(species)} especies x {len(go_columns)} columnas GO\n"
            "(tras el filtro de columnas raras, colsum > 5).",
            fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: pseudo-count ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Paso 1 — Pseudo-count (+1)")
        formula_block(ax, 0.83, r"$count' = count + 1$")
        body_text(ax, 0.74,
            "El siguiente paso es un logaritmo, y log(0) no está definido. Sumar 1 a cada conteo\n"
            "(incluyendo los que ya son > 0) evita ese problema sin necesitar un caso especial para\n"
            "los ceros -- simplemente se desplaza toda la escala en 1 antes de tomar logaritmos.")
        y = example_table(ax, 0.58,
            ("", go_id_zero, go_id_nonzero),
            [
                ("count       ", f"{stages['0. Conteo crudo'][1]:.0f}", f"{stages['0. Conteo crudo'][2]:.0f}"),
                ("count + 1   ", f"{stages['1. + pseudo-count (+1)'][1]:.0f}", f"{stages['1. + pseudo-count (+1)'][2]:.0f}"),
            ],
            col_x=(0.07, 0.42, 0.68))
        body_text(ax, y - 0.03,
            "Importante: un conteo real de 0 se convierte en 1, y un conteo real de 1 también se\n"
            "convertiría en 2 -- ambos casos siguen siendo distinguibles después del pseudo-count.\n"
            "Lo que nunca ocurre es que un 0 y un 1 originales terminen indistinguibles entre sí.",
            fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: log ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Paso 2 — Logaritmo")
        formula_block(ax, 0.83, r"$\log(count + 1)$")
        body_text(ax, 0.74,
            "El logaritmo comprime los valores grandes y expande los valores pequeños: reduce la\n"
            "influencia desproporcionada de GO terms con conteos extremos (colas largas) y hace la\n"
            "distribución mucho más simétrica -- es el paso que más reduce el skew (ver el informe\n"
            "de distribuciones, pca_transform_steps.png).")
        y = example_table(ax, 0.58,
            ("", go_id_zero, go_id_nonzero),
            [
                ("count + 1        ", f"{stages['1. + pseudo-count (+1)'][1]:.0f}", f"{stages['1. + pseudo-count (+1)'][2]:.0f}"),
                ("log(count+1)     ", f"{stages['2. log(count+1)'][1]:.4f}", f"{stages['2. log(count+1)'][2]:.4f}"),
            ],
            col_x=(0.07, 0.42, 0.68))
        body_text(ax, y - 0.03,
            "Nótese que log(1) = 0: todo GO ausente (conteo real 0) queda con el mismo valor (0) en\n"
            "esta etapa, sea cual sea la especie -- la diferenciación entre especies para ese GO\n"
            "vendrá del paso siguiente (CLR), que centra por la media de cada fila.",
            fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: CLR ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Paso 3 — CLR (centered log-ratio)")
        formula_block(ax, 0.83, r"$CLR_i = \log(count_i + 1) - \overline{\log(count + 1)}_{\,fila}$")
        body_text(ax, 0.73,
            "Se resta, a cada valor de la fila (especie), la media de log(count+1) de esa misma fila\n"
            "-- calculada sobre TODAS las columnas GO de esa especie, no solo las dos del ejemplo.\n"
            "Esto elimina el efecto de \"tamaño\" por especie (especies con más proteínas anotadas\n"
            "tienen sistemáticamente valores más altos en casi todas las columnas): tras el CLR, lo\n"
            "que queda es cuánto se desvía cada GO term de lo que es \"típico\" para esa especie, no\n"
            "su magnitud absoluta.")
        body_text(ax, 0.52,
            f"Media real de log(count+1) en la fila de {target_species}\n"
            f"(calculada sobre las {len(go_columns)} columnas GO de esa especie):\n\n"
            f"     media_fila = {row_mean_species:.4f}",
            fontsize=11.5)
        y = example_table(ax, 0.36,
            ("", go_id_zero, go_id_nonzero),
            [
                ("log(count+1)   ", f"{stages['2. log(count+1)'][1]:.4f}", f"{stages['2. log(count+1)'][2]:.4f}"),
                ("- media_fila   ", f"- {row_mean_species:.4f}", f"- {row_mean_species:.4f}"),
                ("= CLR          ", f"{stages['3. CLR (centrado por fila)'][1]:.4f}", f"{stages['3. CLR (centrado por fila)'][2]:.4f}"),
            ],
            col_x=(0.07, 0.42, 0.68))
        body_text(ax, y - 0.03,
            "El GO ausente (conteo 0) queda con un valor NEGATIVO tras el CLR (por debajo de la\n"
            "media de la fila) en vez de en un \"0 universal\" compartido por todas las especies --\n"
            "esto es justamente lo que elimina el efecto de escala por especie.",
            fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 5: StandardScaler ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Paso 4 — StandardScaler (z-score por columna)")
        formula_block(ax, 0.83, r"$z = \dfrac{CLR - \mu_{col}}{\sigma_{col}}$")
        body_text(ax, 0.72,
            "Ahora se estandariza cada COLUMNA (cada GO term) por separado: se resta la media de esa\n"
            "columna (a través de todas las especies) y se divide por su desviación estándar. Así\n"
            "todos los GO terms entran en la PCA con la misma escala -- sin esto, GO terms con más\n"
            "varianza natural dominarían la PCA independientemente de si son biológicamente\n"
            "relevantes.")
        body_text(ax, 0.52,
            f"Media y desviación estándar reales de cada columna, calculadas sobre las\n"
            f"{len(species)} especies del dataset:\n\n"
            f"  {go_id_zero}:      media_col = {col_mean_zero:.4f}   sd_col = {col_std_zero:.4f}\n"
            f"  {go_id_nonzero}:   media_col = {col_mean_nonzero:.4f}   sd_col = {col_std_nonzero:.4f}",
            fontsize=11)
        y = example_table(ax, 0.34,
            ("", go_id_zero, go_id_nonzero),
            [
                ("CLR            ", f"{stages['3. CLR (centrado por fila)'][1]:.4f}", f"{stages['3. CLR (centrado por fila)'][2]:.4f}"),
                ("- media_col    ", f"- {col_mean_zero:.4f}", f"- {col_mean_nonzero:.4f}"),
                ("/ sd_col       ", f"/ {col_std_zero:.4f}", f"/ {col_std_nonzero:.4f}"),
                ("= z-score      ", f"{stages['4. StandardScaler (z-score)'][1]:.4f}", f"{stages['4. StandardScaler (z-score)'][2]:.4f}"),
            ],
            col_x=(0.07, 0.42, 0.68))
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 6: SVD / PCA ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Paso 5 — TruncatedSVD (proyección final)")
        formula_block(ax, 0.83, r"$PC_k = Z \cdot v_k$")
        body_text(ax, 0.73,
            "Este último paso ya no actúa columna a columna: cada especie es un vector fila completo\n"
            "de z-scores (uno por cada GO term retenido), y se proyecta sobre las direcciones (v1, v2,\n"
            "...) que capturan la mayor varianza conjunta entre especies. Por eso no se puede mostrar\n"
            "como una operación sobre una sola celda -- el PC1/PC2 de una especie depende de TODAS\n"
            f"sus {len(go_columns)} columnas GO a la vez, ponderadas por los \"loadings\" (v_k) de cada\n"
            "GO term.")
        body_text(ax, 0.50,
            f"Resultado real para la especie de ejemplo, {target_species}:\n\n"
            f"     PC1 = {species_pc1:.2f}\n"
            f"     PC2 = {species_pc2:.2f}\n\n"
            f"Varianza explicada por todo el dataset en esta proyección:\n\n"
            f"     PC1 = {explained[0]:.1f}%    PC2 = {explained[1]:.1f}%",
            fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 7: dataset-wide summary stats per stage ---
        fig, ax = new_page(pdf)
        title_block(ax, 0.93, "Cómo evoluciona el dataset completo",
                    "Estadísticos globales (todas las especies x todas las columnas GO) en cada etapa")
        headers = ["Etapa", "min", "max", "mean", "mediana", "sd"]
        col_x = [0.07, 0.40, 0.52, 0.64, 0.76, 0.88]
        y = 0.80
        for h, x in zip(headers, col_x):
            ax.text(x, y, h, fontsize=10.5, fontweight="bold", va="top")
        y -= 0.03
        ax.plot([0.07, 0.96], [y + 0.012, y + 0.012], color=GREY, linewidth=0.8, transform=ax.transAxes)
        for label, (st, _, _) in stages.items():
            ax.text(col_x[0], y, label, fontsize=10, va="top")
            ax.text(col_x[1], y, f"{st['min']:.2f}", fontsize=10, va="top", family="monospace")
            ax.text(col_x[2], y, f"{st['max']:.2f}", fontsize=10, va="top", family="monospace")
            ax.text(col_x[3], y, f"{st['mean']:.2f}", fontsize=10, va="top", family="monospace")
            ax.text(col_x[4], y, f"{st['median']:.2f}", fontsize=10, va="top", family="monospace")
            ax.text(col_x[5], y, f"{st['sd']:.2f}", fontsize=10, va="top", family="monospace")
            y -= 0.045
        body_text(ax, y - 0.04,
            "Lectura rápida:\n"
            "  · El rango (min-max) del conteo crudo es enorme (miles de unidades de diferencia).\n"
            "  · Tras el log, ese rango se comprime radicalmente.\n"
            "  · Tras el CLR, la media global queda prácticamente en 0 (centrado por fila).\n"
            "  · Tras StandardScaler, la sd global es 1 por construcción -- cada columna aporta\n"
            "    el mismo peso de partida a la PCA.",
            fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
