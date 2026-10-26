<p align="center">
  <img src="assets/banner.png" width="100%" alt="TCM-AI-Assistant - AI for compound screening and herbal discovery">
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.11-3572A5?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11"></a>
  <a href="#workflow"><img src="https://img.shields.io/badge/Chemprop-2.2-197D80?style=flat-square" alt="Chemprop 2.2"></a>
  <a href="https://doi.org/10.5281/zenodo.13763953"><img src="https://img.shields.io/badge/Dataset-TCM--MKG-66866A?style=flat-square" alt="Dataset: TCM-MKG"></a>
</p>

<p align="center">
  <a href="#workflow">Workflow</a> &nbsp; | &nbsp;
  <a href="#quick-start">Quick start</a> &nbsp; | &nbsp;
  <a href="#reproduce-results">Reproduce results</a> &nbsp; | &nbsp;
  <a href="#data">Data</a>
</p>

## TCM-AI-Assistant

An AI-assisted research workflow for traditional Chinese medicine, connecting target-linked molecular data, positive-unlabeled learning, compound screening, and medicinal-material ranking.

## Workflow

| Module | What it does |
| :--- | :--- |
| [`data/`](data/) | Prepare molecular labels and random or scaffold splits |
| [`modeling/`](modeling/) | Train PU ensembles and predict compound scores |
| [`analysis/`](analysis/) | Evaluate models, rank medicinal materials, and reproduce figures |

**Models:** CAP-MPNN and eight descriptor-based baselines. Predictions average ten final-epoch learners; medicinal materials are ranked by their mean linked-compound score.

## Quick start

Inputs: a compound pool with `InChIKey,SMILES` (or `smiles`), target links with `InChIKey,EntrezID`, and a text file of selected Entrez IDs. Tables can be CSV or TSV; material links and catalogs use TCM-MKG D9/D6 columns.

```bash
python3 -m pip install 'chemprop==2.2.0' 'scikit-learn==1.7.0' pandas matplotlib scipy pillow

bash run.sh prepare compounds.csv compound_targets.tsv targets.txt work/data
bash run.sh train work/data/molecules.csv work/random
bash run.sh evaluate work/random
bash run.sh predict work/random work/data/compounds.csv work/compound_scores.csv
bash run.sh rank work/compound_scores.csv material_compounds.tsv materials.tsv work/material_ranking.csv
```

For scaffold evaluation:

```bash
bash run.sh train work/data/molecules.csv work/scaffold scaffold
bash run.sh evaluate work/scaffold
```

An existing canonical `smiles,label` table can go directly to `train` (`1` positive, `0` unlabeled). Append `mpnn` or `baselines` after the split argument to train one model family; the default is `all`. From the repository root, `python3 main.py` can replace `bash run.sh`.

Splits are saved and reused. Random holdouts use independent PU partitions for MPNN and baselines; scaffold runs share membership. Evaluation outputs go to the run's `evaluation/` directory. Ranking retains repeated associations and stable tie order.

## Reproduce results

Use the external `data` directory from the accompanying processed data package to regenerate manuscript results from archived scores:

```bash
bash run.sh reproduce /path/to/data /path/to/results
```

**Outputs:** Table 1, Tables S1/S3, and nine result panels (Fig. 1a-f and Fig. S5a-c).

## Data

Built on [TCM-MKG](https://doi.org/10.5281/zenodo.13763953) | [Dataset paper](https://doi.org/10.1016/j.jpha.2025.101342). Datasets and model checkpoints are stored separately from this repository.
