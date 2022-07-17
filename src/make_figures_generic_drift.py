import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constants import *
from lib.certify import CertifyGramian
from lib.certify.certify_utils import compute_scores
from lib.utils.plot_utils import init_style

########################################################################################################################
# formatting
FONTSIZE = 18
LINEWIDTH = 1.0
COLORS = init_style(font_size_base=FONTSIZE, linewdith_base=LINEWIDTH, sns_style='whitegrid')

########################################################################################################################
# Labels
GRAMIAN_LOWER = 'Gramian Lower Bd.'
GRAMIAN_UPPER = 'Gramian Upper Bd.'
UPPER_BOUND_MARKERS = 'v'
LOWER_BOUND_MARKERS = '^'

XLABEL = r'Hellinger Distance $\rho$'
YLABELS = {
    JSD_LOSS: 'JSD Loss',
    CLASSIFICATION_ERROR: 'Classification Error',
    CLASSIFICATION_ACCURACY: 'Classification Accuracy',
    AUC_SCORE: 'AUC Score'
}

EMP_SCORE_LABEL = r'$\mathbb{E}_P[\ell(h(X),\,Y)]$'

# init seed
SEED = 742
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def main(data_files, xlims1, ylims1, xlims2, ylims2, titles, save_figure=None):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 5), sharex=True)  # noqa

    # distances
    hellinger_distances = np.linspace(0, 1, 20)

    for j in range(4):
        # load data
        data = np.load(data_files[j], allow_pickle=True)[()]
        labels = data['labels'].astype(int)
        logits = data['logits']

        ################################################################################################################
        # 01-loss
        empirical_01 = compute_scores(logits, labels, func=CLASSIFICATION_ERROR, reduce=REDUCE_MEAN)

        # run certification
        certify_gramian = CertifyGramian(logits, labels, func=CLASSIFICATION_ERROR, finite_sampling=True)
        gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)
        gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)

        # make plot
        axs[0, j].plot(hellinger_distances, gramian_upper_bounds, label=GRAMIAN_UPPER, linestyle='-', lw=2.5, c='black',
                       marker=UPPER_BOUND_MARKERS)
        axs[0, j].plot(hellinger_distances, gramian_lower_bounds, label=GRAMIAN_LOWER, linestyle='-', lw=2.5, c='black',
                       marker=LOWER_BOUND_MARKERS)
        axs[0, j].plot([0], empirical_01, marker='D', color=COLORS[2], lw=2.0, zorder=20)
        axs[0, j].hlines(y=empirical_01, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=11, ls='--',
                         lw=2.0)
        axs[0, j].set_xlim(xlims1[j])
        axs[0, j].set_ylim(ylims1[j])
        axs[0, j].set_title(titles[j])

        if j == 0:
            axs[0, j].set_ylabel(YLABELS[CLASSIFICATION_ERROR])

        ################################################################################################################
        # JSD-loss
        empirical_jsd = compute_scores(logits, labels, func=JSD_LOSS, reduce=REDUCE_MEAN)

        # run certification
        certify_gramian = CertifyGramian(logits, labels, func=JSD_LOSS, finite_sampling=True)
        gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)
        gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)

        # make plot
        axs[1, j].plot(hellinger_distances, gramian_upper_bounds, linestyle='-', lw=2.5, c='black',
                       marker=UPPER_BOUND_MARKERS)
        axs[1, j].plot(hellinger_distances, gramian_lower_bounds, linestyle='-', lw=2.5, c='black',
                       marker=LOWER_BOUND_MARKERS)
        axs[1, j].plot([0], empirical_jsd, marker='D', color=COLORS[2], lw=2.0, zorder=20)
        axs[1, j].hlines(y=empirical_jsd, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=11,
                         ls='--', lw=2.0)
        axs[1, j].set_xlim(xlims2[j])
        axs[1, j].set_ylim(ylims2[j])
        axs[1, j].set_xlabel(XLABEL)

        if j == 0:
            axs[1, j].set_ylabel(YLABELS[JSD_LOSS])

    # legend
    handles = [Line2D([0], [0], color=COLORS[2], marker='D', ls='--', lw=2.0),
               Line2D([0], [0], color='black', marker=UPPER_BOUND_MARKERS, ls='', lw=2.0),
               Line2D([0], [0], color='black', marker=LOWER_BOUND_MARKERS, ls='', lw=2.0)]
    labels = [EMP_SCORE_LABEL, GRAMIAN_UPPER, GRAMIAN_LOWER]
    fig.legend(handles, labels, ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.05))

    # adjust
    plt.subplots_adjust(left=0.1,
                        bottom=0.3,
                        right=0.9,
                        top=0.9,
                        wspace=0.15,
                        hspace=0.2)
    fig.tight_layout()

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.savefig(save_figure.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_figure}')
        return

    plt.show()


if __name__ == '__main__':
    main(data_files=[
        'results/data/imagenet/efficientnet_b7/test-predictions.npy',
        'results/data/cifar10/densenet121/test-predictions.npy',
        'results/data/yelp/BERT_logits/test-predictions.npy',
        'results/data/snli/DeBERTa_logits/test-predictions.npy'
    ],
        xlims1=[(-0.01, 0.2), (-0.01, 0.2), (-0.01, 0.2), (-0.01, 0.2)],
        ylims1=[(-0.02, 0.45), (-0.02, 0.45), (-0.02, 0.7), (-0.02, 0.3)],
        xlims2=[(-0.01, 0.2), (-0.01, 0.2), (-0.01, 0.2), (-0.01, 0.2)],
        ylims2=[(0.07, 0.45), (-0.02, 0.4), (0.05, 0.52), (-0.02, 0.27)],
        titles=['ImageNet-1k', 'CIFAR-10', 'Yelp', 'SNLI'],
        save_figure=f'./results/figures/generic-drift/generic-all.pdf'
    )
