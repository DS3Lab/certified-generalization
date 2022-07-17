import numpy as np
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
GRAMIAN = 'Gramian Certificate'
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
SEED = 742

# init seed
np.random.seed(SEED)
random.seed(SEED)


def main(save_figure=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # distances
    hellinger_distances = np.linspace(0, 0.2, 20)

    ####################################################################################################################
    # IMAGENET
    data_file = 'results/data/binary_imagenet/imagenet_binary_renset152/test-predictions.npy'
    data = np.load(data_file, allow_pickle=True)[()]
    labels = data['labels'].astype(int)
    logits = data['logits']
    empirical_auc = compute_scores(logits, labels, func=AUC_SCORE, reduce=REDUCE_MEAN)

    # run certification
    certify_gramian = CertifyGramian(logits, labels, func=AUC_SCORE, finite_sampling=True)
    gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)
    gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)

    # make plot
    axs[0].plot(hellinger_distances, gramian_upper_bounds, label=GRAMIAN, linestyle='-', lw=2.5, c='black',
                marker=UPPER_BOUND_MARKERS, markevery=4)
    axs[0].plot(hellinger_distances, gramian_lower_bounds, linestyle='-', lw=2.5, c='black',
                marker=LOWER_BOUND_MARKERS, markevery=4)
    axs[0].plot([0], empirical_auc, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    axs[0].hlines(y=empirical_auc, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=11, ls='--',
                  lw=2.0)
    axs[0].set_xlim((-0.01, 0.2))
    axs[0].set_ylim((0.55, 1.02))
    axs[0].set_title('ImageNet-1k')
    axs[0].set_ylabel(YLABELS[AUC_SCORE])
    axs[0].set_xlabel(XLABEL)

    ####################################################################################################################
    # CIFAR10
    data_file = 'results/data/binary_cifar10/cifar_resnet110_binary/test-predictions.npy'
    data = np.load(data_file, allow_pickle=True)[()]
    labels = data['labels'].astype(int)
    logits = data['logits']
    empirical_auc = compute_scores(logits, labels, func=AUC_SCORE, reduce=REDUCE_MEAN)

    # run certification
    certify_gramian = CertifyGramian(logits, labels, func=AUC_SCORE, finite_sampling=True)
    gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)
    gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)

    ####################################################################################################################
    # make plot
    axs[1].plot(hellinger_distances, gramian_upper_bounds, label=GRAMIAN, linestyle='-', lw=2.5, c='black',
                marker=UPPER_BOUND_MARKERS, markevery=4)
    axs[1].plot(hellinger_distances, gramian_lower_bounds, linestyle='-', lw=2.5, c='black',
                marker=LOWER_BOUND_MARKERS, markevery=4)
    axs[1].plot([0], empirical_auc, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    axs[1].hlines(y=empirical_auc, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=11, ls='--',
                  lw=2.0)
    axs[1].set_xlim((-0.01, 0.2))
    axs[1].set_ylim((0.55, 1.02))
    axs[1].set_title('CIFAR-10')
    axs[1].set_xlabel(XLABEL)

    # legend
    handles = [Line2D([0], [0], color=COLORS[2], marker='D', ls='--', lw=2.0),
               Line2D([0], [0], color='black', marker=UPPER_BOUND_MARKERS, ls='-', lw=2.0),
               Line2D([0], [0], color='black', marker=LOWER_BOUND_MARKERS, ls='-', lw=2.0)]
    labels = [EMP_SCORE_LABEL, 'Gramian Upper Bd.', 'Gramian Lower Bd.']
    axs[1].legend(handles, labels, ncol=1, frameon=True, loc='lower left', fontsize=FONTSIZE - 2, framealpha=1.0,
                  fancybox=False)

    fig.tight_layout()

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.savefig(save_figure.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_figure}')
        return

    plt.show()


if __name__ == '__main__':
    main(save_figure='./results/figures/generic-drift/generic-auc.pdf')
    # main()
