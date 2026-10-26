"""Build a canonical, positive-unlabeled molecular pool from target links."""
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem


def read_table(path, **kwargs):
    return pd.read_csv(path, sep='\t' if Path(path).suffix == '.tsv' else ',', **kwargs)


def canonical_smiles(value):
    if not isinstance(value, str) or not value.strip():
        return None
    mol = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None


def canonical_pool(compounds):
    compounds = compounds.rename(columns={'SMILES': 'smiles'}).copy()
    compounds['smiles'] = compounds.smiles.map(canonical_smiles)
    compounds = compounds.dropna(subset=['InChIKey', 'smiles'])
    compounds = compounds[['InChIKey', 'smiles']].drop_duplicates()
    if compounds.InChIKey.duplicated().any():
        raise ValueError('Each InChIKey must map to one canonical structure.')
    return compounds.reset_index(drop=True)


def prepare(compounds_path, links_path, targets_path, output):
    compounds = canonical_pool(read_table(compounds_path))
    targets = set(Path(targets_path).read_text(encoding='utf-8-sig').split())
    positive = set()
    for links in read_table(links_path, usecols=['InChIKey', 'EntrezID'],
                            dtype=str, chunksize=500000):
        positive.update(links.loc[links.EntrezID.isin(targets), 'InChIKey'])
    compounds['label'] = compounds.InChIKey.isin(positive).astype(int)
    # Aliases of the same structure inherit a positive label from any matched key.
    compounds['label'] = compounds.groupby('smiles').label.transform('max')
    molecules = compounds[['smiles', 'label']].drop_duplicates('smiles')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(output / 'molecules.csv', index=False)
    compounds.to_csv(output / 'compounds.csv', index=False)
    print(f'Prepared {len(molecules)} molecules; {molecules.label.sum()} positive labels.')


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('Usage: python3 prepare.py COMPOUNDS LINKS TARGETS OUTPUT_DIRECTORY')
    prepare(*sys.argv[1:])
