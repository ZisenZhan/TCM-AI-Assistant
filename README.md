# CAP-MPNN reproduction

Reproduce Table 1, Tables S1/S3, and Fig. 1a-f / Fig. S5a-c from saved prediction scores.

```bash
python3 -m pip install numpy pandas matplotlib scipy scikit-learn pillow
bash run.sh /path/to/data /path/to/results
```

Use the `data` folder from the accompanying processed data package. The output path is optional and defaults to `results/`.

Source dataset: [TCM-MKG](https://doi.org/10.5281/zenodo.13763953) ([paper](https://doi.org/10.1016/j.jpha.2025.101342)).
