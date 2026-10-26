"""Train ten-member PU ensembles on a saved random or scaffold split."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data.prepare import canonical_smiles
from data.splits import MEMBERS, make_splits


def train(input_path, output, mode='random', family='all'):
    if mode not in ('random', 'scaffold') or family not in ('all', 'mpnn', 'baselines'):
        raise ValueError('Use random/scaffold and all/mpnn/baselines.')
    output = Path(output)
    molecules = pd.read_csv(input_path)[['smiles', 'label']]
    molecules['smiles'] = molecules.smiles.map(canonical_smiles)
    if molecules.smiles.isna().any() or molecules.smiles.duplicated().any():
        raise ValueError('Training requires valid, unique canonical structures.')
    if set(molecules.label) != {0, 1}:
        raise ValueError('Labels must be 1 (positive) and 0 (unlabeled).')
    molecules['label'] = molecules.label.astype(int)
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / 'splits.npz'
    if split_path.exists():
        if not molecules.equals(pd.read_csv(output / 'molecules.csv')):
            raise ValueError('This run directory belongs to a different molecular pool.')
        with np.load(split_path, allow_pickle=False) as saved:
            split = dict(saved)
        if str(split['mode']) != mode:
            raise ValueError('Use a separate run directory for a different split.')
    else:
        split = make_splits(molecules, mode)
        molecules.to_csv(output / 'molecules.csv', index=False)
        np.savez_compressed(split_path, **split)
    test = split['test']
    labels = molecules.label.to_numpy()
    if family in ('all', 'mpnn'):
        from modeling import mpnn
        scores, losses = [], []
        for member in range(MEMBERS):
            folder = output / f'member_{member:02d}'
            folder.mkdir(exist_ok=True)
            score_file = folder / 'mpnn_scores.npy'
            if not score_file.exists():
                print(f'CAP-MPNN: member {member + 1}/{MEMBERS}', flush=True)
                score = mpnn.train_member(molecules, split[f'mpnn_train_{member}'],
                    split[f'mpnn_val_{member}'], test, folder)
                np.save(score_file, score)
            scores.append(np.load(score_file, allow_pickle=False))
            losses.append(pd.read_csv(folder / 'loss.csv').assign(Member=member + 1))
        np.savez_compressed(output / 'mpnn_scores.npz', models=np.array(['CAP-MPNN']),
                            member_scores=np.array(scores)[None, :, :])
        pd.concat(losses, ignore_index=True).to_csv(output / 'training_loss.csv', index=False)
    if family in ('all', 'baselines'):
        from modeling.baselines import DescriptorFilter, descriptors, train_member, build_models
        matrix, names = descriptors(molecules.smiles)
        # Random-split filtering uses development data; scaffold filtering is per member.
        shared_filter = DescriptorFilter().fit(matrix[split['development']]) if mode == 'random' else None
        scores = []
        with threadpool_limits(limits=2):
            for member in range(MEMBERS):
                folder = output / f'member_{member:02d}'
                folder.mkdir(exist_ok=True)
                score_file = folder / 'baseline_scores.npy'
                if not score_file.exists():
                    print(f'Baselines: member {member + 1}/{MEMBERS}', flush=True)
                    ids = split[f'baseline_train_{member}']
                    preprocessing = shared_filter or DescriptorFilter().fit(matrix[ids])
                    bundle, score = train_member(matrix, labels, ids, test, preprocessing)
                    bundle['descriptor_names'] = names
                    joblib.dump(bundle, folder / 'baselines.joblib', compress=3)
                    np.save(score_file, score)
                scores.append(np.load(score_file, allow_pickle=False))
        np.savez_compressed(output / 'baseline_scores.npz', models=np.array(list(build_models())),
                            member_scores=np.stack(scores, axis=1))


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4, 5):
        raise SystemExit('Usage: python3 main.py train MOLECULES RUN_DIRECTORY [random|scaffold] [all|mpnn|baselines]')
    train(*sys.argv[1:])
