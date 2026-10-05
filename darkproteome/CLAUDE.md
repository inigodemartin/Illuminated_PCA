# Dark proteome — working context

Session knowledge for `PCA/darkproteome/`. Read this before touching anything here.

## What this is
The "dark proteome" = proteins that homology-based annotation (AHRD) leaves without any GO term. FANTASIA (embedding-based) annotates many of them. We compare, across 2672 species, the GO profile of dark proteins vs. proteins both tools annotate.

## Data (results/, not in git, ~120 MB per matrix)
Built on the server by `PCA/scripts/dark_proteome_matrices.py`, cleaned locally by `PCA/scripts/filter_dark_proteome_results.py`. **Always use the `*_clean.tsv` versions.**

| File | Content |
|---|---|
| `mod01_fantasia_only_counts__clean.tsv` | Species × GO counts. Proteins annotated by FANTASIA but with no homology GO (**the dark proteome**). GO from FANTASIA. |
| `mod01_fantasia_both_counts__clean.tsv` | Same, proteins annotated by both tools (control, same GO source). GO from FANTASIA. |
| `mod01_homology_all_counts__clean.tsv` | Whole proteome, GO from homology (AHRD). Classical reference. |
| `mod02_dark_proteome_stats__clean.tsv` | Per-species stats comparing the two groups (coverage, length, GO richness, IC, FANTASIA/homology agreement). |
| `mod03_taxons__clean.tsv` | `Group`/`Species`, only viridi / non_viridi. |
| `excluded_species_.tsv` | 17 broken species removed (8 plants with FANTASIA IDs absent from AHRD = different proteome version; 4 fungi with 1-protein FANTASIA file; 5 species with 14–103-protein AHRD table). |

Matrix facts: 2672 species (2091 non_viridi, 581 viridi), first column = species directory name, header = GO ids, integer counts. 21855 / 23349 / 24561 GO columns respectively. "Annotated by homology" = AHRD GO column non-empty (BLAST hit without GO does not count). Cell = number of proteins of that species, in that group, carrying that GO.

Key findings already reported (2026-10-02): dark proteome ≈ 37 % of the proteome (plants 31.5 %, non-plants 38.8 %); dark proteins ~150 aa shorter and 4× more often < 100 aa; GO/protein and IC equal or slightly higher in the dark group; 29 % of dark-group GO terms absent from the same species' `both` group, growing with dark-proteome size (rho 0.61); FANTASIA/homology Jaccard on shared proteins ≈ 0.10.

## PCAs (done 2026-10-05)
Run with IkusiGO v0.14.0 (`/home/inigo/claude/IkusiGO`, pushed, commit 9bdafa0), using its new `--matrix` mode (added this session so these matrices can be fed directly; streams the TSV into int32, `--manifest` optional and only `Species`/`TaxID` read, in-place PCA transforms to fit in this 3 GB WSL, peak ≈ 1.8 GB, < 1 min per matrix).

Outputs in `pca_fantasia_only/`, `pca_fantasia_both/`, `pca_homology_all/` → `results/mod03_pca_{presence_absence,abundance}_pca_<matrix>[_custom_taxonomy].html` + `_top_loadings.tsv` / `_full_loadings.tsv`. Two colourings each:
- plain = NCBI kingdom (2482/2672 resolved; manifest `PCA/data/species_taxid.tsv`)
- `_custom_taxonomy` = `PCA/species_taxonomy.tsv` (Fungi 1904, angiosperms 492, Protists 170, chlorophyta 49, bryophytes 25, Rhodophyta 14, gymnosperms 8, lycophytes 5, pteridophyte 2, Unclassified 3). Same coordinates, only legend differs.

Command (from the IkusiGO repo; add `--taxonomy /home/inigo/claude/PCA/species_taxonomy.tsv` for the custom colouring; `--skip_genome_stats` avoids ~15 min of NCBI queries):
```bash
python3 scripts/ikusigo.py --matrix <results>/mod01_<m>_counts__clean.tsv \
    --manifest /home/inigo/claude/PCA/data/species_taxid.tsv \
    --output <darkproteome>/pca_<m> --skip_genome_stats
```

Explained variance PC1/PC2/PC3 (%):

| Matrix | Abundance (CLR) | Presence/absence |
|---|---|---|
| fantasia_only | 30.7 / 8.6 / 3.3 | 9.1 / 3.6 / 2.7 |
| fantasia_both | 41.3 / 7.0 / 4.1 | 18.2 / 5.3 / 3.2 |
| homology_all | 40.9 / 7.4 / 3.0 | 11.3 / 6.1 / 3.7 |

Reading: the dark proteome has less dominant structure (PC1 ~10 points lower); `both` and homology look alike in abundance, so FANTASIA recovers the same taxonomic structure on shared proteins.

Three species have no taxid and no taxonomy row (strain-suffixed names): `Exophiala_dermatitidis_NIH_UT8656`, `Fomitiporia_mediterranea_MF3_22`, `Phaeodactylum_tricornutum_CCAP_1055_1` → "Unclassified". Fix by adding them to `species_taxonomy.tsv` / `species_taxid.tsv` and re-running the PCA only.

`species_taxonomy.tsv` has 7 duplicated species with conflicting Metazoa/Protists groups (choanoflagellates etc.); none are in these matrices.

## Pending (plan agreed 2026-10-02)
1. ~~PCAs per clean matrix~~ done.
2. ~~Compare `fantasia_only` vs `both`: GO profile, functional enrichment~~ done (see below).
3. Check on the server whether the 8 ID-mismatch plants can be recovered (Claude has no server access; Q&A only).

## Enrichment: dark proteome vs both (done 2026-10-05)
Script `PCA/scripts/dark_proteome_enrichment.py` (v0.1.0), outputs in `enrichment/results/` (not in git). Rerun (~2 min, 1.4 GB peak):
```bash
python3 scripts/dark_proteome_enrichment.py \
    --only  darkproteome/results/mod01_fantasia_only_counts__clean.tsv \
    --both  darkproteome/results/mod01_fantasia_both_counts__clean.tsv \
    --stats darkproteome/results/mod02_dark_proteome_stats__clean.tsv \
    --taxonomy species_taxonomy.tsv --output darkproteome/enrichment --format png,pdf
```
Method: species are the replicates. Per GO term, fraction of proteins carrying it in the dark group vs the both group **of the same species**; per-species log2FC (pseudocount 0.5), median across species, fraction of species with the term more frequent in dark, paired Wilcoxon across species + BH FDR. Call = FDR < 0.05, |median log2FC| ≥ 1, >50 % species in that direction. Tested only if present in ≥ 50 species. Stratified medians per Group (viridi/non_viridi) and per `species_taxonomy.tsv` group (≥ 30 species: Fungi 1904, angiosperms 492, Protists 170, chlorophyta 49, other 57).

| File | Content |
|---|---|
| `mod01_term_enrichment_*.tsv` | 23 980 GO terms (union of both matrices): 18 316 tested, 3363 enriched in dark, 4500 depleted; 631 terms never seen in any species' both group. Volcano + `mod01_top_terms_*` barplot. |
| `mod02_goslim_profile_*.tsv` | Annotation-level profile on `goslim_pir` (434 terms, 97 % of annotations mapped; `goslim_generic` left 36 % unmapped — no cytoplasm/protein binding/response to stress). Dumbbell plot. |
| `mod03_dark_characteristics_*.tsv` | mod02 stats summarised per stratum (medians, paired Wilcoxon). Boxplot panel. |

Findings:
- **Depleted in dark = the conserved housekeeping core** homology captures well: ribosome / cytoplasmic translation (log2FC −4), TCA, ATP/GTP binding & hydrolysis, oxidoreductases, transporters, amino-acid/lipid/organic-acid metabolism, ER, vacuole, mitochondrial matrix, splicing, rRNA processing. Slim: catalytic activity 8.1 % vs 11.7 % of annotations, metabolic process 8.6 % vs 13.2 %, transport 1.7 % vs 3.4 %.
- **Enriched in dark = regulation / binding / adaptor functions**, consistent in fungi, plants and protists: protein binding (6.5 % vs 3.5 %), molecular adaptor, enzyme regulator, kinase binding/activator, ubiquitin-ligase adaptors + SCF complex, post-transcriptional regulation of gene expression (2.1 % vs 0.65 %), transcription regulator, DNA binding, chromosome segregation/kinetochore, cilium/axoneme, cell septum, extracellular space, glutathione transferase/peroxidase. Enriched terms carry 32 % of dark annotations but only 7 % of both annotations.
- **Caveat — implausible transfers**: a layer of very specific metazoan/viral/host–pathogen terms is enriched in ~2000 species including fungi and plants at ~0.1–0.9 % of dark proteins each: host cell nucleus/cytoplasm, viral process (slim log2FC +3.4), leukotriene-C4 synthase, endothelin/VEGF receptor, complement binding, chemokine binding, granulosa cell proliferation, syncytial embryo cellularization… 2971/3363 enriched terms are near-absent (< 0.02 %) in the both group. Likely FANTASIA embedding neighbours from animal/viral UniProt entries landing on short lineage-specific proteins (effector-like). Treat high-IC enriched terms with caution; the mid-level (slim) picture is the robust one.
- Characteristics (n = 2672, per-species medians): length 278 vs 435 aa (shorter in 100 % of species), < 100 aa 8.4 % vs 1.9 %, GO/protein 5.08 vs 5.24, mean IC 13.4 vs 13.3 (higher in 79 % of species), BP/MF/CC shares ≈ equal (40/28/32 %), dark share of proteome 35.7 %, 29 % of dark GO terms absent from both, FANTASIA/homology Jaccard 0.086.

## False positives: GO taxon-constraint check (done 2026-10-05)
Script `PCA/scripts/dark_proteome_taxon_check.py` (v0.1.0), outputs in `taxon_check/results/` (not in git). Needs the three matrices (dark, both, homology); ~2.7 min, **2.1 GB peak** (close to the WSL ceiling — close other things first). Rerun:
```bash
python3 scripts/dark_proteome_taxon_check.py \
    --only  darkproteome/results/mod01_fantasia_only_counts__clean.tsv \
    --both  darkproteome/results/mod01_fantasia_both_counts__clean.tsv \
    --homology darkproteome/results/mod01_homology_all_counts__clean.tsv \
    --stats darkproteome/results/mod02_dark_proteome_stats__clean.tsv \
    --taxonomy species_taxonomy.tsv --output darkproteome/taxon_check --format png,pdf
```
Reference data (committed, built by `scripts/build_go_taxon_constraints.py` from go-edit.obo + go-taxon-groupings.obo + NCBI taxdump, 2026-10-05): `data/go_taxon_constraints.tsv` (1658 asserted only_in/never_in constraints on 1380 GO terms — they live only in go-edit.obo / go-plus.owl, NOT in go.obo or go-basic), `data/go_taxon_unions.tsv` (15 union taxa such as "Fungi or Bacteria"), `data/species_lineage_taxids.tsv` (full NCBI ancestor taxids for the 3906 species of species_lineage.tsv with a TaxID). Constraints are inherited through is_a + part_of (8553/27693 GO columns end up constrained); a species violates when its lineage lacks the only_in taxon or contains the never_in taxon. 2667/2672 species have a lineage.

**Module 1 — % of GO annotations violating a GO taxon constraint (per-species medians):**

| Group | dark | both | homology | dark>both |
|---|---|---|---|---|
| all (2667) | 7.09 | 3.35 | 0.55 | 99 % of species |
| Fungi (1902) | 7.40 | 3.44 | 0.63 | 100 % |
| angiosperms (492) | 3.84 | 1.60 | 0.21 | 99 % |
| Protists (170) | 7.69 | 6.46 | 1.16 | 95 % |
| chlorophyta (49) | 7.36 | 5.42 | 0.92 | 100 % |

Distinct GO terms violating: 11.4 % dark, 8.2 % both, 3.5 % homology. Ratio dark/both ≈ 2.0 per species, dark/homology ≈ 12. The dark violation rate grows with the dark share of the proteome (Spearman 0.42). Split: only_in 3.4 % + never_in 3.9 % (dark). Biggest offending constraints: only_in Metazoa (23 % of dark violations), never_in Fungi (22 %), only_in Eumetazoa (11 %), never_in Ascomycota (6 %), only_in Arthropoda (5 %). Top offending terms (dark annotations): ciliary plasm (never_in Ascomycota; 174k), kinetoplast (only_in Kinetoplastea; 128k), centrosome (never_in Fungi/Viridiplantae; 84k — also the top homology offender, 29k: SPB→centrosome transfer), regulation of neuronal synaptic plasticity, P granule, spermatogenesis, behavioral response to ethanol, synapse, viral tegument/capsid (never_in cellular organisms), skeletal system morphogenesis, long-term memory, complement binding / C-X-C chemokine binding (never_in Fungi; 0 by homology).

**Module 2 — clade-unsupported terms** (GO never assigned by homology to any species of the clade; soft, includes real novelty): median % of dark annotations on such terms Fungi 3.7 (both 1.1), angiosperms 4.6 (1.4), Protists 6.3 (3.5), chlorophyta 9.3 (5.0). Top in Fungi: symbiont-mediated suppression of host NF-κB, RING-like zinc finger domain binding, 2',3'-cGAMP binding, host cell nuclear envelope, chemokine activity, complement binding; in angiosperms: RING-like zinc finger domain binding (161k), RNR inhibitor activity, muscle system process, chemokine activity, pheromone activity, ergosterol regulation.

Reading: the official constraints alone flag ~7 % of dark annotations (and 11 % of dark terms) as biologically impossible for the organism, twice the rate of FANTASIA on homology-supported proteins and 12× the homology baseline. This is a lower bound: terms without a taxon constraint (e.g. "host cell nucleus", "leukotriene-C4 synthase activity" in a fungus) are not caught; Module 2 covers part of that. Limits: annotation-level counts (not proteins); a few violations may be legitimate (HGT, outdated constraints); the 2026 constraints are applied to the 2025 go-basic hierarchy.

## Taxon-constraint check on the whole-proteome FANTASIA matrix (2026-10-05)
Same script in single-matrix mode (v0.2.0: `--matrix` + `--label`, Module 2 auto-skipped) on `merged_PCA_belen_fantasia.tsv` (3107 species x 29 724 GO, FANTASIA counts for the whole proteome), stratified by `species_taxonomy.tsv`. Asgard species have no NCBI taxid, so `build_go_taxon_constraints.py --group_fallback Asgard=1935183` (Promethearchaeati) gives them the group-level lineage Asgard → Archaea → cellular organisms; 3101/3107 species evaluated. Output `taxon_check_belen_fantasia/results/` (not in git). Rerun (~1.5 min, 1.5 GB):
```bash
python3 scripts/dark_proteome_taxon_check.py --matrix merged_PCA_belen_fantasia.tsv --label fantasia \
    --taxonomy species_taxonomy.tsv --min_group_species 2 \
    --output darkproteome/taxon_check_belen_fantasia --format png,pdf
```
Median % of FANTASIA annotations violating a GO taxon constraint, per group (only_in / never_in split; % distinct terms):

| Group | n | % annotations | only_in | never_in | % terms |
|---|---|---|---|---|---|
| Asgard | 436 | 18.0 | 16.5 | 2.4 | 16.8 |
| Glaucophyta | 3 | 8.6 | 8.1 | 1.2 | 11.1 |
| Protists | 220 | 7.3 | 7.1 | 0.3 | 9.6 |
| Rhodophyta | 36 | 7.1 | 6.7 | 0.2 | 9.2 |
| bryophytes | 25 | 6.5 | 5.6 | 0.8 | 9.2 |
| chlorophyta | 49 | 6.2 | 5.1 | 0.9 | 9.2 |
| lycophytes | 5 | 5.6 | 5.1 | 0.7 | 8.9 |
| Fungi | 847 | 4.6 | 2.4 | 2.7 | 9.9 |
| gymnosperms | 8 | 4.6 | 3.8 | 0.9 | 8.9 |
| Metazoa | 961 | 2.5 | 1.8 | 0.6 | 4.7 |
| angiosperms | 509 | 2.3 | 1.7 | 0.6 | 7.4 |

Reading: the false-positive rate tracks the distance to the well-annotated model organisms that dominate GO/UniProt. Metazoa and angiosperms (2.3–2.5 %) are the reference clades; fungi double that; protists, red/green algae and bryophytes ~3×; Asgard archaea 18 %, almost entirely `only_in Eukaryota` terms (nucleus 1.6 % of all their annotations, mitochondrion 1.3 %, ER, nucleoplasm, peroxisome, Golgi…). Overall top constraints: only_in Metazoa (15 % of violations), Eumetazoa, Arthropoda, never_in cellular organisms (viral terms: virion membrane / envelope / capsid / tegument / nucleocapsid, each in > 2600 species — 7 %), never_in Fungi, only_in Vertebrata. Per-group top offenders are in `mod01_taxon_violations_terms_by_group_*.tsv` (new in v0.2.0): Metazoa → viral terms, inflammatory response (only_in Vertebrata, in 878 invertebrates), chloroplast; angiosperms → ciliary plasm, centrosome, synaptic terms, kinetoplast; Fungi → ciliary plasm (Ascomycota), spermatogenesis, embryo development, kinetoplast, P granule; Protists/algae → kinetoplast (only_in Kinetoplastea, in every species), metazoan development/synapse terms, "embryo development ending in seed dormancy".
Consistency check: Fungi here (whole proteome) 4.6 % sits between the dark (7.4 %) and both (3.4 %) values of the dark-proteome dataset, as expected for the mixture.

## Rules
- Memory: 3 GB WSL. Never `pd.read_csv` these matrices naively (OOM); stream or use int32. Run PCAs sequentially.
- PCA/ repo: commit + push directly. IkusiGO repo: confirm with the user before committing.
- Respond in Spanish; file contents in English.
