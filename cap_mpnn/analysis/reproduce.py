"""Reproduce the computational tables and panels from saved prediction scores."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from cap_mpnn.analysis.evaluate import comparison_table, full_table, evaluation_frames, loss_frame
from cap_mpnn.analysis.plots import draw_panels
from cap_mpnn.analysis.rank import rank_materials


def analyse(data):
    with np.load(data / 'predictions.npz', allow_pickle=False) as z:
        models = z['models'].tolist()
        y, scores = z['heldout_labels'], z['heldout_scores']
        sy, ss = z['scaffold_labels'], z['scaffold_scores']
        subsets = z['subsample_positions']
    frames, context = evaluation_frames(y, scores, models)
    frames['Table1'] = comparison_table(y, scores, models, subsets)
    frames['TableS3'] = full_table(sy, ss, models)
    associations = pd.read_csv(data / 'candidate_associations.csv', float_precision='round_trip')
    catalog = pd.read_csv(data / 'candidate_materials.csv', keep_default_na=False)
    ranked = rank_materials(associations, catalog)
    frames['TableS1'] = ranked.head(10)[['Rank', 'Database_label', 'Mean_score', 'Selected', 'Selection_rationale']]
    frames['Fig1ab'] = pd.read_csv(data / 'resource_counts.csv')
    rows = []
    with np.load(data / 'network_degrees.npz', allow_pickle=False) as z:
        for panel, key in [('Fig1c', 'material_degree'), ('Fig1d', 'target_degree')]:
            degree, counts = np.unique(z[key], return_counts=True)
            rows.extend({'Panel': panel, 'Degree': int(d), 'Count': int(n), 'Probability': n / len(z[key])}
                        for d, n in zip(degree, counts))
    frames['Fig1cd'] = pd.DataFrame(rows)
    frames['FigS5c'] = loss_frame(data / 'training_loss.csv')
    return frames, context


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: python3 -m cap_mpnn reproduce DATA_DIRECTORY [OUTPUT_DIRECTORY]')
    data = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) == 3 else Path('results')
    frames, context = analyse(data)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('Table1', 'TableS1', 'TableS3'):
        frames[name].to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
    draw_panels(frames, context, output)
    print('Reproduced 3 tables and 9 result panels from the saved scores.')


if __name__ == '__main__':
    main()
