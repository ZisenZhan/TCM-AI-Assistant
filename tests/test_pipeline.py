from pathlib import Path
import random
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from cap_mpnn.modeling.baselines import DescriptorFilter, train_member
from cap_mpnn.analysis.evaluate import comparison_table, expected_hits, metrics
from cap_mpnn.data.prepare import canonical_pool, prepare
from cap_mpnn.analysis.rank import rank_materials, rank
from cap_mpnn.data.splits import MEMBERS, make_splits, scaffold_split


class PipelineTests(unittest.TestCase):
    def test_prepare_merges_structure_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({'InChIKey': ['a', 'b', 'c'],
                          'SMILES': ['CCO', 'OCC', 'CCN']}).to_csv(root / 'pool.csv', index=False)
            pd.DataFrame({'InChIKey': ['b', 'c'], 'EntrezID': ['gene-a', 'gene-b']}).to_csv(
                root / 'links.tsv', sep='\t', index=False)
            (root / 'targets.txt').write_text('gene-a\n', encoding='utf-8')
            prepare(root / 'pool.csv', root / 'links.tsv', root / 'targets.txt', root / 'out')
            molecules = pd.read_csv(root / 'out/molecules.csv')
            aliases = pd.read_csv(root / 'out/compounds.csv')
            self.assertEqual(dict(zip(molecules.smiles, molecules.label)), {'CCO': 1, 'CCN': 0})
            self.assertEqual(aliases.label.tolist(), [1, 1, 0])

    def test_conflicting_structure_keys_rejected(self):
        with self.assertRaises(ValueError):
            canonical_pool(pd.DataFrame({'InChIKey': ['a', 'a'], 'smiles': ['CCO', 'CCN']}))

    def test_random_pu_memberships(self):
        labels = np.tile([1, 0, 0, 0], 400)
        mol = pd.DataFrame({'smiles': np.arange(len(labels)).astype(str), 'label': labels})
        split = make_splits(mol, 'random')
        development, test = set(split['development']), set(split['test'])
        self.assertFalse(development & test)
        self.assertEqual(development | test, set(range(len(labels))))
        positive = {i for i in development if labels[i] == 1}
        for family in ('mpnn', 'baseline'):
            seen = set()
            for member in range(MEMBERS):
                train, val = (set(split[f'{family}_{part}_{member}']) for part in ('train', 'val'))
                pool = train | val
                self.assertFalse(train & val or pool & test)
                self.assertTrue(pool <= development)
                self.assertEqual({i for i in pool if labels[i] == 1}, positive)
                unlabeled = {i for i in pool if labels[i] == 0}
                self.assertFalse(seen & unlabeled)
                seen |= unlabeled
            self.assertEqual(seen, development - positive)

    def test_scaffolds_do_not_cross_holdouts(self):
        smiles = [f'C1{"C" * (size - 1)}1' for size in range(3, 33) for _ in range(20)]
        train, val, test = scaffold_split(smiles, random.Random())
        scaffolds = [set(MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles[i]) for i in ids)
                     for ids in (train, val, test)]
        self.assertFalse(scaffolds[0] & scaffolds[1] or scaffolds[0] & scaffolds[2] or scaffolds[1] & scaffolds[2])
        self.assertEqual(sorted(np.r_[train, val, test]), list(range(len(smiles))))
        split = make_splits(pd.DataFrame({'smiles': smiles, 'label': np.tile([0, 1], len(smiles) // 2)}), 'scaffold')
        for member in range(MEMBERS):
            np.testing.assert_array_equal(split[f'mpnn_train_{member}'], split[f'baseline_train_{member}'])
            np.testing.assert_array_equal(split[f'mpnn_val_{member}'], split['validation'])

    def test_descriptor_filter_uses_training_only(self):
        train = np.array([[1., np.nan, 10., 4.], [2., np.nan, 20., 4.],
                          [3., np.nan, 30., 4.], [4., np.nan, 40., 4.]])
        fitted = DescriptorFilter().fit(train)
        self.assertEqual(fitted.transform(train).shape, (4, 1))
        heldout = fitted.transform(np.array([[999., 5., 9999., 9.], [np.nan, 1., np.nan, 2.]]))
        self.assertAlmostEqual(heldout[0, 0], fitted.upper[0])
        self.assertAlmostEqual(heldout[1, 0], 2.5)

    def test_all_baselines_train_and_reload(self):
        matrix = np.random.default_rng().normal(size=(160, 6))
        labels = np.tile([0, 1], 80)
        train, test = np.arange(120), np.arange(120, 160)
        preprocessing = DescriptorFilter().fit(matrix[train])
        bundle, scores = train_member(matrix, labels, train, test, preprocessing)
        self.assertEqual(scores.shape, (8, len(test)))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'models.joblib'
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            features = loaded['scaler'].transform(loaded['descriptor_filter'].transform(matrix[test]))
            restored = np.array([model.predict_proba(features)[:, 1] for model in loaded['models'].values()])
            np.testing.assert_array_equal(scores, restored)

    def test_tied_scores_and_trapezoidal_auprc(self):
        labels = np.tile([0, 1], 100)
        scores = np.ones(len(labels))
        self.assertEqual(expected_hits(labels, scores, 25), 12.5)
        np.testing.assert_allclose(metrics(labels, scores), [.5, .75, 1.])
        with self.assertRaises(ValueError):
            expected_hits(labels, scores, len(labels) + 1)

    def test_subsamples_use_shared_membership_and_sample_sd(self):
        labels = np.r_[np.ones(100, dtype=int), np.zeros(200, dtype=int)]
        scores = np.linspace(0, 1, len(labels))
        subsets = np.array([np.arange(100, 120), np.arange(150, 170), np.arange(280, 300)])
        table = comparison_table(labels, [scores, scores], ['A', 'B'], subsets)
        np.testing.assert_array_equal(table.iloc[0, 1:].to_numpy(), table.iloc[1, 1:].to_numpy())
        values = np.array([metrics(labels[np.r_[np.arange(100), s]], scores[np.r_[np.arange(100), s]])
                           for s in subsets])
        self.assertAlmostEqual(table.iloc[0].Down_AUPRC_SD, values[:, 1].std(ddof=1))

    def test_ranking_keeps_duplicate_associations_and_stable_ties(self):
        associations = pd.DataFrame({'Material': ['a', 'a', 'a', 'b', 'c'],
                                     'Score': [1., 1., 0., 2 / 3, np.nan]})
        ranked = rank_materials(associations, pd.DataFrame({'Material': ['b', 'a', 'c', 'd']}))
        self.assertEqual(ranked.Material.tolist(), ['b', 'a'])
        np.testing.assert_allclose(ranked.Mean_score, [2 / 3, 2 / 3])
        self.assertEqual(ranked.Rank.tolist(), [1, 2])

    def test_raw_material_links_join_many_to_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({'InChIKey': ['a', 'b'], 'Score': [1., 0.]}).to_csv(root / 'scores.csv', index=False)
            pd.DataFrame({'CHP_ID': ['h', 'h', 'h'], 'InChIKey': ['a', 'a', 'b']}).to_csv(
                root / 'links.tsv', sep='\t', index=False)
            pd.DataFrame({'CHP_ID': ['h', 'i']}).to_csv(root / 'catalog.csv', index=False)
            rank(root / 'scores.csv', root / 'links.tsv', root / 'catalog.csv', root / 'rank.csv')
            result = pd.read_csv(root / 'rank.csv')
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result.Mean_score.iloc[0], 2 / 3)


if __name__ == '__main__':
    unittest.main()
