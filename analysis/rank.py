"""Rank medicinal materials by the mean of their linked compound scores."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def rank_materials(associations, catalog):
    means = associations.groupby('Material', sort=False)['Score'].mean()
    ranked = catalog.copy()
    ranked['Mean_score'] = ranked.Material.map(means)
    ranked = ranked.dropna(subset=['Mean_score'])
    ranked = ranked.sort_values('Mean_score', ascending=False, kind='stable')
    ranked['Rank'] = np.arange(1, len(ranked) + 1)
    return ranked


def rank(scores_path, links_path, catalog_path, output):
    from data.prepare import read_table
    scores = read_table(scores_path, usecols=['InChIKey', 'Score'], float_precision='round_trip')
    links = read_table(links_path, usecols=['CHP_ID', 'InChIKey'])
    catalog = read_table(catalog_path).rename(columns={'CHP_ID': 'Material'})
    if catalog.Material.duplicated().any():
        raise ValueError('The material catalog must have one row per CHP_ID.')
    associations = links.merge(scores, on='InChIKey', how='inner', validate='many_to_one')
    associations = associations.rename(columns={'CHP_ID': 'Material'})
    ranked = rank_materials(associations, catalog)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False, float_format='%.17g')


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('Usage: python3 main.py rank SCORES MATERIAL_LINKS MATERIAL_CATALOG OUTPUT_CSV')
    rank(*sys.argv[1:])
