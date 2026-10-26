from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cap_mpnn.modeling.mpnn import predict_checkpoints, train_member


class MPNNTests(unittest.TestCase):
    def test_training_and_final_checkpoint_prediction(self):
        smiles = ['C' * length + tail for length in range(2, 9)
                  for tail in ('O', 'N', 'Cl', 'Br', 'F', 'C(=O)O')]
        molecules = pd.DataFrame({'smiles': smiles, 'label': np.tile([0, 1], len(smiles) // 2)})
        train, val, test = np.arange(30), np.arange(30, 36), np.arange(36, 42)
        with tempfile.TemporaryDirectory() as temp:
            scores = train_member(molecules, train, val, test, temp, epochs=3)
            history = pd.read_csv(Path(temp) / 'loss.csv')
            self.assertEqual(history.Epoch.tolist(), [0, 1, 2])
            self.assertTrue(np.isfinite(history[['Training_loss', 'Validation_loss']]).all().all())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            checkpoint = Path(temp) / 'mpnn.ckpt'
            restored = predict_checkpoints([checkpoint, checkpoint], molecules.smiles.iloc[test])
            np.testing.assert_allclose(scores, restored, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
