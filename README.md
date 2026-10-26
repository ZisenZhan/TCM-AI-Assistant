# CAP-MPNN

Target-linked positive-unlabeled learning for compound screening and medicinal-material ranking: CAP-MPNN, eight descriptor baselines, random/scaffold evaluation, and manuscript figure reproduction.

Source dataset: [TCM-MKG](https://doi.org/10.5281/zenodo.13763953) ([paper](https://doi.org/10.1016/j.jpha.2025.101342)). This repository contains code only.

```bash
python3 -m pip install 'chemprop==2.2.0' 'scikit-learn==1.7.0' pandas matplotlib scipy pillow
```

## Train and screen

Inputs: a modeling compound pool with `InChIKey,SMILES` (or `smiles`); compound-target links with `InChIKey,EntrezID`; and a text file of selected Entrez IDs. Tables may be CSV or TSV. Material links/catalog use the TCM-MKG D9/D6 columns.

```bash
bash run.sh prepare compounds.csv compound_targets.tsv targets.txt work/data
bash run.sh train work/data/molecules.csv work/random
bash run.sh evaluate work/random
bash run.sh predict work/random work/data/compounds.csv work/compound_scores.csv
bash run.sh rank work/compound_scores.csv material_compounds.tsv materials.tsv work/material_ranking.csv

bash run.sh train work/data/molecules.csv work/scaffold scaffold
bash run.sh evaluate work/scaffold
```

An existing canonical `smiles,label` table can go directly to `train` (`1` positive, `0` unlabeled). Training averages ten final-epoch learners. Splits are saved and reused; random holdouts use independent PU partitions for MPNN and baselines, while scaffold runs share membership. Optional final `train` argument: `mpnn` or `baselines` (default `all`).

Evaluation writes metrics, curve data and figures under the run's `evaluation/` directory. Ranking averages linked scores, retaining repeated associations and stable tie order.

## Reproduce manuscript results

Use the `data` directory from the accompanying processed data package. This route uses archived scores, not newly trained models.

```bash
bash run.sh reproduce /path/to/data /path/to/results
python3 -m unittest discover -s tests
```

Outputs: Table 1, Tables S1/S3 and nine result panels (Fig. 1a-f and Fig. S5a-c).
