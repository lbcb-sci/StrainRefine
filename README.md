# StrainRefine

**StrainRefine** is a method for refining strain-level metagenomic classification by analyzing read–reference mapping profiles.  
It reduces false-positive strain detections by grouping highly similar reference genomes and consolidating their read support.

The method can be used as a **standalone post-mapping refinement step** or **integrated into the MADRe pipeline** as a replacement for the original `ReadClassification` step.

---

# Overview

[![StrainRefine workflow](https://github.com/lbcb-sci/StrainRefine/raw/master/figures/method_overview.png)](figures/method_overview.png)

Strain-level metagenomic classification is challenging because reads originating from a single strain often map equally well to several closely related reference genomes. This can lead to redundant detections and large numbers of false-positive strain calls.

StrainRefine addresses this problem by analyzing **read–reference mapping profiles**:

- constructing binary read-support profiles for candidate reference genomes
- measuring similarity between references based on shared mapped reads
- clustering highly similar references within species
- filtering weakly supported references
- reassigning reads to representative genomes

This consolidation prevents multiple near-identical genomes from being reported separately and helps recover the underlying strain signal even when many highly similar genomes are present in the reference database.

---

# Integration with MADRe

StrainRefine was originally developed as a refinement step within the [**MADRe**](https://github.com/lbcb-sci/MADRe) pipeline.

MADRe consists of two main stages:

1. **Database reduction** using metagenomic assembly and contig-to-reference mapping
2. **Read classification** against the reduced reference database

StrainRefine is designed to **replace the original MADRe `ReadClassification` step**.  
Instead of assigning reads directly to candidate reference genomes, StrainRefine analyzes read–reference mapping profiles to identify groups of highly similar references and refine read assignments.

Although developed for MADRe, StrainRefine can also be used as a **standalone post-mapping refinement step** for pipelines that produce **read–reference mappings in PAF format**.

---

# Requirements

- Python 3.8+
- Python packages:
  * `numpy`
  * `scikit-learn`

Install dependencies using:

```
pip install numpy scikit-learn
```

## Input

StrainRefine requires the following inputs.

### 1. PAF file

Read-to-reference mappings in **PAF format** (e.g. produced by `minimap2`).

Example: `--paf_path mappings.paf`

### 2. Strain–species mapping

JSON file mapping **strain taxids to species taxids**. The file can be found in `database/`

Example: `--strain_species_info taxids_species.json`

---

## Usage

Basic usage:

```
python StrainRefine.py \
    --paf_path mappings.paf \
    --strain_species_info taxids_species.json
```

## Parameters

### Required arguments

| Parameter | Description |
| --- | --- |
| `--paf_path` | Path to the PAF file containing read–reference mappings. |
| `--strain_species_info` | JSON file mapping strain taxids to species taxids. If using the MADRe database, use `MADRe/database/taxids_species.json`. |

---

### Clustering

| Parameter | Description | Default |
| --- | --- | --- |
| `--cluster_eps` | DBSCAN epsilon value used for clustering references within species. | `0.8` |

---

### Reference support thresholds

| Parameter | Description | Default |
| --- | --- | --- |
| `--high_score_threshold` | Score threshold used to count high-confidence assigned reads per reference. | `0.6` |
| `--min_high_score_reads` | Minimum number of high-score reads required for a reference to be considered supported. | `5` |
| `--mid_high_score_reads` | Intermediate support threshold for references. | `10` |
| `--min_high_score_fraction` | Minimum fraction of high-score reads required for moderately supported references. | `0.8` |

---

### Species-level thresholds *(currently informative)*

| Parameter | Description | Default |
| --- | --- | --- |
| `--species_min_read_count` | Minimum read count threshold for species reliability. | `5` |
| `--species_min_mean_score` | Minimum mean score threshold for species reliability. | `0.6` |
| `--species_low_count_cap` | Species with mean score below threshold and read count below this cap are treated as unreliable. | `30` |

---

### Output

| Parameter | Description | Default |
| --- | --- | --- |
| `--read_class_output` | Path to the output file containing read classification labels. | `read_classification.out` |
| `--clustering_out` | Directory for clustering-related output files. | `clustering_output/` |
| `--ref_summary_output` | Path to the TSV file containing per-reference read counts and horizontal coverage. | `reference_summary.tsv` |

## Output

StrainRefine produces refined read classifications, clustering information, and a per-reference summary.

### Read classification

| File | Description |
| --- | --- |
| `read_classification.out` | Final strain-level classification labels for reads after refinement. |

Each line contains a single read and its assigned reference genome:

```
read_id : reference_name
```

---

### Reference summary

| File | Description |
| --- | --- |
| `reference_summary.tsv` | Per-reference summary with read count and horizontal coverage, sorted by read count in descending order. |

This is a tab-separated file with a header line. Each subsequent line corresponds to one final reference genome:

```
reference    read_count    horizontal_coverage
|511145|NC_000913.3   4821    0.923400
|83334|NC_002695.2     312     0.187200
```

- **reference** — reference genome identifier as it appears in the PAF file
- **read_count** — number of reads assigned to this reference after all refinement and reassignment steps
- **horizontal_coverage** — fraction of the reference genome length covered by at least one mapped read (0.0–1.0)

---

### Clustering output

| Directory | Description |
| --- | --- |
| `clustering_output/` | Directory containing clustering results for reference genomes. |

The directory contains two files:

| File | Description |
| --- | --- |
| `clusters.txt` | Each line represents one cluster and contains reference genome names belonging to that cluster, separated by spaces. |
| `representatives.txt` | Each line contains the representative reference genome selected for the corresponding cluster. |

The **line order corresponds between the two files**, meaning that the representative listed on line *i* in `representatives.txt` corresponds to the cluster listed on line *i* in `clusters.txt`.