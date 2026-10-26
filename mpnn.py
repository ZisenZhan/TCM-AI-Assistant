"""Directed-bond CAP-MPNN training and final-checkpoint prediction."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, BinaryClassificationFFN


def loader(smiles, labels=None, shuffle=False):
    targets = labels if labels is not None else [None] * len(smiles)
    points = [MoleculeDatapoint.from_smi(s, None if y is None else np.array([y], dtype=float))
              for s, y in zip(smiles, targets)]
    dataset = MoleculeDataset(points, SimpleMoleculeMolGraphFeaturizer())
    dataset.cache = True
    return build_dataloader(dataset, batch_size=64, num_workers=0, shuffle=shuffle)


class History(pl.Callback):
    def __init__(self):
        self.rows = []
        self.loss_sum = 0.
        self.count = 0

    def on_train_epoch_start(self, trainer, model):
        self.loss_sum, self.count = 0., 0

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        n = len(batch.Y)
        self.loss_sum += float(loss.detach().cpu()) * n
        self.count += n

    def on_train_epoch_end(self, trainer, model):
        self.rows.append({'Epoch': trainer.current_epoch,
            'Training_loss': self.loss_sum / self.count,
            'Validation_loss': float(trainer.callback_metrics['val_loss'].detach().cpu())})


def trainer(**kwargs):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, **kwargs)


def predict_model(model, data_loader):
    batches = trainer().predict(model, data_loader)
    return torch.cat(batches).detach().cpu().numpy().reshape(-1)


def train_member(molecules, train_ids, val_ids, test_ids, output, epochs=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = MPNN(BondMessagePassing(d_v=72, d_e=14, d_h=300, depth=3), MeanAggregation(),
                 BinaryClassificationFFN(n_tasks=1, input_dim=300, hidden_dim=300, n_layers=1),
                 batch_norm=False, warmup_epochs=min(2, epochs - 1),
                 init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
    history = History()
    fit = trainer(max_epochs=epochs, callbacks=[history], num_sanity_val_steps=0)
    fit.fit(model, loader(molecules.smiles.iloc[train_ids], molecules.label.iloc[train_ids], True),
            loader(molecules.smiles.iloc[val_ids], molecules.label.iloc[val_ids]))
    fit.save_checkpoint(output / 'mpnn.ckpt')
    pd.DataFrame(history.rows).to_csv(output / 'loss.csv', index=False)
    return predict_model(model, loader(molecules.smiles.iloc[test_ids]))


def predict_checkpoints(paths, smiles):
    data_loader = loader(smiles)
    scores = []
    for path in paths:
        model = MPNN.load_from_checkpoint(path, map_location='cpu')
        scores.append(predict_model(model, data_loader))
    return np.mean(scores, axis=0)
