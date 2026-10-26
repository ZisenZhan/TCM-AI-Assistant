"""Average the ten final CAP-MPNN checkpoints for compound screening."""
from pathlib import Path
import sys

import pandas as pd

from cap_mpnn.modeling.mpnn import predict_checkpoints
from cap_mpnn.data.prepare import canonical_pool, read_table
from cap_mpnn.data.splits import MEMBERS


def predict(run, compounds_path, output):
    paths = [Path(run) / f'member_{i:02d}' / 'mpnn.ckpt' for i in range(MEMBERS)]
    compounds = canonical_pool(read_table(compounds_path))
    structures = compounds.smiles.drop_duplicates().tolist()
    scores = pd.Series(predict_checkpoints(paths, structures), index=structures)
    compounds['Score'] = compounds.smiles.map(scores)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compounds.to_csv(output, index=False, float_format='%.17g')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python3 -m cap_mpnn predict RUN_DIRECTORY COMPOUNDS OUTPUT_CSV')
    predict(*sys.argv[1:])
