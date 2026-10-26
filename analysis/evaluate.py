"""Evaluate fixed ensemble scores with trapezoidal AUPRC and tie-aware EF100."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


METRICS = ['AUROC', 'AUPRC', 'EF100']


def expected_hits(labels, scores, k):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not 1 <= k <= len(scores):
        raise ValueError('Top-K cannot exceed the evaluation-set size.')
    cutoff = np.partition(scores, len(scores) - k)[-k]
    above, tied = scores > cutoff, scores == cutoff
    return float(labels[above].sum() + (k - above.sum()) * labels[tied].mean())


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Metrics require aligned finite scores and both binary label classes.')
    precision, recall, _ = precision_recall_curve(labels, scores)
    return [roc_auc_score(labels, scores), auc(recall, precision),
            expected_hits(labels, scores, 100) / (100 * labels.mean())]


def comparison_table(labels, scores, models, subsets):
    full = np.array([metrics(labels, s) for s in scores])
    positive = np.flatnonzero(labels == 1)
    samples = np.array([[metrics(labels[idx], s[idx]) for s in scores]
                        for chosen in subsets for idx in [np.r_[positive, chosen]]])
    averages, deviations = samples.mean(axis=0), samples.std(axis=0, ddof=1)
    rows = []
    for i, model in enumerate(models):
        row = {'Model': 'Gradient boosting' if model == 'GB' else model}
        row.update({f'Full_{name}': float(full[i, j]) for j, name in enumerate(METRICS)})
        for j, name in enumerate(METRICS):
            row[f'Down_{name}_mean'] = float(averages[i, j])
            row[f'Down_{name}_SD'] = float(deviations[i, j])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('Full_AUROC', kind='stable')


def full_table(labels, scores, models):
    return pd.DataFrame([
        {'Model': 'Gradient boosting' if name == 'GB' else name,
         **dict(zip(METRICS, metrics(labels, s)))} for name, s in zip(models, scores)
    ]).sort_values('AUROC', kind='stable')


def evaluation_frames(labels, scores, models):
    frames = {'Fig1e': pd.DataFrame({'K': np.arange(1, 101), **{
        model: [expected_hits(labels, s, k) / (k * labels.mean()) for k in range(1, 101)]
        for model, s in zip(models, scores)}})}
    context = {'models': models, 'labels': labels,
               'full_metrics': np.array([metrics(labels, s) for s in scores])}
    if 'CAP-MPNN' in models:
        focal = scores[models.index('CAP-MPNN')]
        context['medians'] = [float(np.median(focal[labels == value])) for value in (0, 1)]
        if all(np.std(focal[labels == value]) > 0 for value in (0, 1)):
            grid = np.linspace(0, 1, 501)
            frames['Fig1f'] = pd.DataFrame({'Score': grid,
                'Unlabeled_density': gaussian_kde(focal[labels == 0])(grid),
                'Positive_density': gaussian_kde(focal[labels == 1])(grid)})
    roc, pr = [], []
    for model, s in zip(models, scores):
        fpr, tpr, _ = roc_curve(labels, s)
        roc.extend({'Model': model, 'FPR': a, 'TPR': b} for a, b in zip(fpr, tpr))
        precision, recall, _ = precision_recall_curve(labels, s, drop_intermediate=True)
        pr.extend({'Model': model, 'Recall': a, 'Precision': b} for a, b in zip(recall, precision))
    frames['FigS5a'], frames['FigS5b'] = pd.DataFrame(roc), pd.DataFrame(pr)
    return frames, context


def loss_frame(path):
    raw = pd.read_csv(path, float_precision='round_trip')
    loss = raw.groupby('Epoch')[['Training_loss', 'Validation_loss']].agg(['mean', 'std'])
    loss.columns = ['Training_mean', 'Training_SD', 'Validation_mean', 'Validation_SD']
    return loss.reset_index()


def evaluate(run):
    from analysis.plots import draw_panels
    run = Path(run)
    molecules = pd.read_csv(run / 'molecules.csv')
    with np.load(run / 'splits.npz', allow_pickle=False) as split:
        labels = molecules.label.to_numpy()[split['test']]
        mode = str(split['mode'])
    names, scores = [], []
    for family in ('mpnn', 'baseline'):
        path = run / f'{family}_scores.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                names.extend(saved['models'].tolist())
                scores.extend(saved['member_scores'].mean(axis=1))
    if not names:
        raise ValueError('No trained ensemble scores found in this run directory.')
    scores = np.array(scores)
    output = run / 'evaluation'
    output.mkdir(exist_ok=True)
    if mode == 'random':
        subset_path = output / 'subsample_positions.npy'
        if subset_path.exists():
            subsets = np.load(subset_path, allow_pickle=False)
        else:
            unlabeled = np.flatnonzero(labels == 0)
            rng = np.random.default_rng()
            subsets = np.array([rng.choice(unlabeled, len(unlabeled) // 10, replace=False) for _ in range(100)])
            np.save(subset_path, subsets)
        table = comparison_table(labels, scores, names, subsets)
    else:
        table = full_table(labels, scores, names)
    table.to_csv(output / 'metrics.csv', index=False, float_format='%.17g')
    pd.DataFrame({'label': labels, **dict(zip(names, scores))}).to_csv(
        output / 'ensemble_scores.csv', index=False, float_format='%.17g')
    frames, context = evaluation_frames(labels, scores, names)
    if (run / 'training_loss.csv').exists():
        frames['FigS5c'] = loss_frame(run / 'training_loss.csv')
    for name, frame in frames.items():
        frame.to_csv(output / f'{name}.csv', index=False, float_format='%.17g')
    draw_panels(frames, context, output)
    print(f'Evaluated {len(names)} ensembles on {len(labels)} held-out molecules.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 main.py evaluate RUN_DIRECTORY')
    evaluate(sys.argv[1])
