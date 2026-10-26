"""Shared plotting style for saved-score reproduction and new model runs."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text


COLORS = {'CAP-MPNN': '#557F91', 'GB': '#6C9276', 'AdaBoost': '#C58D66',
          'NBAYES': '#B78EA5', 'QDA': '#8DA79A', 'SGD logistic': '#A7AAA5',
          'LDA': '#B18A46', 'Linear SVM': '#926B94', 'Logistic regression': '#747EAE'}
LINES = {'CAP-MPNN': '-', 'GB': (0, (7, 2)), 'AdaBoost': (0, (4, 2, 1, 2)),
         'NBAYES': (0, (2, 2)), 'QDA': (0, (7, 2, 1.5, 2)),
         'SGD logistic': (0, (5, 2)), 'LDA': (0, (2, 2)),
         'Linear SVM': (0, (7, 2, 1.5, 2)), 'Logistic regression': '-'}
MARKERS = {'CAP-MPNN': 'o', 'GB': '<', 'AdaBoost': '>', 'NBAYES': 's', 'QDA': '^',
           'SGD logistic': 'D', 'LDA': 'v', 'Linear SVM': 'P', 'Logistic regression': 'X'}
PIE_COLORS = ['#7299B2', '#A9C1D3', '#C99BAD', '#B7A8C8', '#9FB596', '#D7B48C', '#C9C9C4']


def draw_panels(frames, context, output):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 16,
        'axes.labelsize': 19, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
        'axes.linewidth': 1.15, 'legend.fontsize': 14.5, 'legend.frameon': False,
        'axes.unicode_minus': False, 'text.color': '#202020', 'axes.labelcolor': '#202020'})
    models = context['models']
    output.mkdir(parents=True, exist_ok=True)

    def legend_under(ax, labels=None, markers=False):
        handles = [Line2D([0], [0], color=COLORS[m], linestyle=LINES[m],
            linewidth=2.5 if m == 'CAP-MPNN' else 1.8,
            marker=MARKERS[m] if markers else None, markerfacecolor='white',
            markersize=5, markeredgewidth=1.1) for m in models]
        ax.legend(handles, labels or models, loc='upper center', bbox_to_anchor=(.46, -.20),
            ncol=2, fontsize=13.5, handlelength=1.7, handletextpad=.4,
            columnspacing=.8, labelspacing=.20, borderaxespad=0, borderpad=0)

    def entities(ax):
        counts = frames['Fig1ab'].query("Panel == 'Fig1a'").Count.to_numpy()
        ax.barh(range(4), counts, height=.62, color=['#D9DEE1', '#C9DCE5', '#D9D0E1', '#BDD4CE'],
            edgecolor=['#8E999F', '#7899A9', '#A696B7', '#729C91'], linewidth=1.3)
        ax.set_yticks(range(4), ['Medicinal\nmaterials', 'Linked\ncompounds', 'SMILES\ncompounds', 'Targets'], fontsize=16)
        ax.invert_yaxis()
        for i, value in enumerate(counts):
            ax.text(value + 3500, i, f'{value:,}', va='center', fontsize=16)
        ax.set(xlim=(0, 175000), xlabel='Entity count')
        ax.set_xticks([0, 50000, 100000, 150000], ['0', '50k', '100k', '150k'])

    def composition(ax):
        data = frames['Fig1ab'].query("Panel == 'Fig1b'")
        vals = data.Count.to_numpy()
        wedges, _ = ax.pie(vals, colors=PIE_COLORS, startangle=90, counterclock=False,
            wedgeprops={'width': .48, 'edgecolor': 'white', 'linewidth': 1})
        ax.text(0, 0, f'{vals.sum():,}\nrecords', ha='center', va='center', fontsize=17)
        labels = [f'{name} {value / vals.sum() * 100:.2f}%' for name, value in zip(data.Item, vals)]
        ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(.47, -.08), ncol=2,
            fontsize=14.5, handlelength=.75, handletextpad=.35, columnspacing=.65, labelspacing=.27)
        ax.axis('off')

    def degree(ax, panel, label, color, note):
        data = frames['Fig1cd'].loc[frames['Fig1cd'].Panel.eq(panel)]
        ax.scatter(data.Degree, data.Probability, s=36, color=color, edgecolor='white', linewidth=.4, zorder=5)
        ax.set(xscale='log', yscale='log', xlabel=label, ylabel='Probability')
        ax.set_xticks([1, 100, 10000])
        lo, hi = ax.get_ylim()
        ax.set_yticks([tick for tick in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if lo <= tick <= hi])
        ax.grid(color='#E6E6E4', linewidth=.7)
        ax.text(.97, .97, note, transform=ax.transAxes, ha='right', va='top', fontsize=14.5, color='#555555')
        if panel == 'Fig1d':
            ax.xaxis.label.set_fontsize(17.5)

    def topk(ax):
        data = frames['Fig1e']
        for i, name in enumerate(models):
            ax.plot(data.K, data[name], color=COLORS[name], linestyle=LINES[name],
                linewidth=2.7 if name == 'CAP-MPNN' else 1.8, zorder=10 if name == 'CAP-MPNN' else 5)
            ix = np.arange(5 + i * 4, 100, 73)
            ax.plot(data.K.to_numpy()[ix], data[name].to_numpy()[ix], linestyle='none', marker=MARKERS[name],
                markersize=5.8, markerfacecolor='white', markeredgecolor=COLORS[name], markeredgewidth=1.25, zorder=15)
        ax.axhline(1, color='#B9B7B3', linestyle=(0, (5, 3)), linewidth=1)
        upper = max(15, int(np.ceil(data[models].to_numpy().max() / 5) * 5))
        ax.set(xlim=(0, 100), ylim=(0, upper), xlabel='Top-K', ylabel='Enrichment factor')
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks(np.arange(0, upper + 1, 5))
        ax.grid(axis='y', color='#ECEDEA', linewidth=.7)
        legend_under(ax, markers=True)

    def density(ax):
        data = frames['Fig1f']
        upper = 0
        for i, label, color in [(0, 'Unlabeled', '#7E9FB0'), (1, 'Positive', '#C491A7')]:
            values = data[label + '_density']
            upper = max(upper, values.max())
            ax.plot(data.Score, values, color=color, linewidth=2.25, label=label)
            ax.axvline(context['medians'][i], color=color, linestyle=(0, (4, 3)), linewidth=1.3)
        upper = int(np.ceil(upper * 1.08))
        ax.set(xlim=(0, 1), ylim=(0, upper), xlabel='Ensemble score', ylabel='Density')
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.arange(0, upper + 1))
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.24), fontsize=14.5)

    def curve(ax, kind):
        is_roc = kind == 'FigS5a'
        data = frames[kind]
        x, yy = ('FPR', 'TPR') if is_roc else ('Recall', 'Precision')
        if is_roc:
            ax.plot([0, 1], [0, 1], color='#B9B7B3', linestyle=(0, (5, 3)), linewidth=1)
        else:
            ax.axhline(context['labels'].mean(), color='#B9B7B3', linestyle=(0, (5, 3)), linewidth=1)
        for name in models[::-1]:
            d = data.loc[data.Model.eq(name)]
            ax.plot(d[x], d[yy], color=COLORS[name], linestyle=LINES[name], linewidth=2.6 if name == 'CAP-MPNN' else 1.7)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False positive rate' if is_roc else 'Recall',
               ylabel='True positive rate' if is_roc else 'Precision')
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        column = 0 if is_roc else 1
        legend_under(ax, [f'{name} {context["full_metrics"][i, column]:.2f}' for i, name in enumerate(models)])

    def loss(ax):
        data = frames['FigS5c']
        for key, color, ls in [('Training', '#7E9FB0', '-'), ('Validation', '#C491A7', (0, (5, 3)))]:
            mean, sd = data[key + '_mean'], data[key + '_SD']
            ax.fill_between(data.Epoch, mean - sd, mean + sd, color=color, alpha=.12, linewidth=0)
            ax.plot(data.Epoch, mean, color=color, linestyle=ls, linewidth=2.2, label=key)
        last = int(data.Epoch.max())
        ax.set(xlim=(0, max(1, last)), xlabel='Epoch', ylabel='Loss')
        ax.set_xticks([tick for tick in [0, 5, 10, 15] if tick < last] + [last])
        ax.legend(loc='upper right', fontsize=14.5)

    drawers = [
        ('Fig1a_entities', entities), ('Fig1b_composition', composition),
        ('Fig1c_material_degree', lambda ax: degree(ax, 'Fig1c', 'Material-target degree', '#789FB3', 'n = 2,150 materials')),
        ('Fig1d_target_degree', lambda ax: degree(ax, 'Fig1d', 'Target-compound degree', '#C491A7', 'n = 14,522 targets')),
        ('Fig1e_enrichment', topk), ('Fig1f_score_density', density),
        ('FigS5a_ROC', lambda ax: curve(ax, 'FigS5a')), ('FigS5b_PR', lambda ax: curve(ax, 'FigS5b')),
        ('FigS5c_loss', loss)]
    for name, draw in drawers:
        key = name.split('_')[0]
        key = 'Fig1ab' if key in ('Fig1a', 'Fig1b') else 'Fig1cd' if key in ('Fig1c', 'Fig1d') else key
        if key not in frames:
            continue
        fig = plt.figure(figsize=(5.05, 5.65))
        ax = fig.add_axes([1.38 / 5.05, 1.85 / 5.65, 3.05 / 5.05, 3.05 / 5.65])
        ax.set_box_aspect(1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=5, width=1.1, pad=4)
        ax.set_axisbelow(True)
        draw(ax)
        ax.xaxis.set_zorder(50)
        ax.yaxis.set_zorder(50)
        for text in ax.findobj(Text):
            text.set_zorder(70)
        if ax.get_legend():
            ax.get_legend().set_zorder(100)
        fig.savefig(output / (name + '.tif'), dpi=600, transparent=True,
                    bbox_inches='tight', pad_inches=.1, pil_kwargs={'compression': 'tiff_lzw'})
        plt.close(fig)
