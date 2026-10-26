from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from cap_mpnn.__main__ import COMMANDS


ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_all_commands_resolve(self):
        for module in COMMANDS.values():
            self.assertIsNotNone(importlib.util.find_spec(f'cap_mpnn.{module}'))

    def test_prepare_preserves_external_relative_paths(self):
        with tempfile.TemporaryDirectory(prefix='pipeline test ') as temp:
            folder = Path(temp)
            pd.DataFrame({'InChIKey': ['a', 'b'], 'SMILES': ['CCO', 'CCN']}).to_csv(
                folder / 'pool.csv', index=False)
            pd.DataFrame({'InChIKey': ['a'], 'EntrezID': ['gene-a']}).to_csv(
                folder / 'links.tsv', sep='\t', index=False)
            (folder / 'targets.txt').write_text('gene-a\n', encoding='utf-8')
            subprocess.run([sys.executable, '-B', str(ROOT / 'cap_mpnn/__main__.py'),
                'prepare', 'pool.csv', 'links.tsv', 'targets.txt', 'output folder'],
                cwd=folder, check=True, capture_output=True, text=True)
            result = pd.read_csv(folder / 'output folder/molecules.csv')
            self.assertEqual(result.label.tolist(), [1, 0])

    def test_python_module_entry(self):
        with tempfile.TemporaryDirectory(prefix='pipeline test ') as temp:
            folder = Path(temp)
            pd.DataFrame({'InChIKey': ['a', 'b'], 'Score': [.8, .2]}).to_csv(
                folder / 'scores.csv', index=False)
            pd.DataFrame({'CHP_ID': ['h', 'h'], 'InChIKey': ['a', 'b']}).to_csv(
                folder / 'links.csv', index=False)
            pd.DataFrame({'CHP_ID': ['h']}).to_csv(folder / 'catalog.csv', index=False)
            subprocess.run([sys.executable, '-B', '-m', 'cap_mpnn', 'rank',
                str(folder / 'scores.csv'), str(folder / 'links.csv'),
                str(folder / 'catalog.csv'), str(folder / 'ranking.csv')],
                cwd=ROOT, check=True, capture_output=True, text=True)
            result = pd.read_csv(folder / 'ranking.csv')
            self.assertAlmostEqual(result.Mean_score.iloc[0], .5)


if __name__ == '__main__':
    unittest.main()
