"""Random/scaffold holdouts and ten disjoint unlabeled PU partitions."""
import random

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


MEMBERS = 10


def scaffold_split(smiles, rng):
    groups = {}
    for index, value in enumerate(smiles):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False)
        groups.setdefault(scaffold, []).append(index)
    groups = [groups[key] for key in sorted(groups)]
    n_train, n_val = int(.8 * len(smiles)), int(.1 * len(smiles))
    n_test = len(smiles) - n_train - n_val
    large = [g for g in groups if len(g) > n_test / 2]
    small = [g for g in groups if len(g) <= n_test / 2]
    rng.shuffle(small)
    train = [index for group in large for index in group]
    val, test = [], []
    for group in small:
        dest = train if len(train) + len(group) <= n_train else val if len(val) + len(group) <= n_val else test
        dest.extend(group)
    return tuple(np.asarray(part, dtype=int) for part in (train, val, test))


def pu_members(labels, development, validation, mode, rng, rounding):
    positive = development[labels[development] == 1]
    unlabeled = development[labels[development] == 0]
    permutation = rng.permutation(unlabeled)
    if rounding == 'floor' and mode == 'random':
        width = len(permutation) // MEMBERS
        groups = [permutation[i * width:(i + 1) * width] for i in range(MEMBERS - 1)]
        groups.append(permutation[(MEMBERS - 1) * width:])
    else:
        groups = np.array_split(permutation, MEMBERS)
    result = []
    for group in groups:
        if mode == 'scaffold':
            result.append((np.r_[positive, group], validation))
            continue
        train, val = [], []
        for pool in (positive, group):
            pool = rng.permutation(pool)
            n = int(.9 * len(pool)) if rounding == 'floor' else round(.9 * len(pool))
            n = max(1, min(n, len(pool) - 1))
            train.extend(pool[:n])
            val.extend(pool[n:])
        result.append((np.asarray(train, dtype=int), np.asarray(val, dtype=int)))
    return result


def make_splits(molecules, mode):
    labels = molecules.label.to_numpy()
    if mode == 'random':
        development, test = train_test_split(np.arange(len(labels)), test_size=.2, stratify=labels)
        validation = np.array([], dtype=int)
    elif mode == 'scaffold':
        development, validation, test = scaffold_split(molecules.smiles, random.Random())
    else:
        raise ValueError('Split must be random or scaffold.')
    for indices in (development, test):
        if set(labels[indices]) != {0, 1}:
            raise ValueError('Both label classes are required in development and test sets.')
    if mode == 'scaffold' and set(labels[validation]) != {0, 1}:
        raise ValueError('The scaffold validation set needs both label classes.')
    result = dict(mode=np.array(mode), development=development, validation=validation, test=test)
    mpnn = pu_members(labels, development, validation, mode, np.random.default_rng(), 'floor')
    baseline = mpnn if mode == 'scaffold' else pu_members(
        labels, development, validation, mode, np.random.default_rng(), 'round')
    for family, members in [('mpnn', mpnn), ('baseline', baseline)]:
        for member, (train, val) in enumerate(members):
            if set(labels[train]) != {0, 1} or set(labels[val]) != {0, 1}:
                raise ValueError('The molecular pool is too small for ten PU learners.')
            result[f'{family}_train_{member}'] = train
            result[f'{family}_val_{member}'] = val
    return result
