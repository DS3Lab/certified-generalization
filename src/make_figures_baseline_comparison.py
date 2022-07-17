import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constants import MLP_V2
from lib.utils.plot_utils import init_style
from run_certify_wrm import compute_bounds

# plot params
font_size = 18
linewidth = 1.0
colors = init_style(font_size_base=font_size, linewdith_base=linewidth, sns_style='whitegrid')

USE_CUDA = torch.cuda.is_available()
SEED = 742

XLABEL = r'Distribution shift ($L_2$-norm  $\||\delta\||_2$)'
YLABEL = 'JSD loss'
WRM_CERTIFICATE = 'WRM (Sinha et al., 2018)'
GRAMIAN_METHOD_LEGEND = 'Gramian Certificate'
LIPSCHITZ_CERT = 'Lip. (Cranko et al., 2021)'

# init seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def make_plot_multiple(root_dirs, linestyles, progression_labels, arch=MLP_V2, save_figure_as=None):
    # make figure
    fig = plt.figure(figsize=(8, 5))

    for rd, ls in zip(root_dirs, linestyles):
        # load data
        if any([
            not os.path.exists(os.path.join(rd, 'euclidean_distances.npy')),
            not os.path.exists(os.path.join(rd, 'loss.npy')),
            not os.path.exists(os.path.join(rd, 'wrm_bounds.npy')),
            not os.path.exists(os.path.join(rd, 'lipschitz_bounds.npy')),
            not os.path.exists(os.path.join(rd, 'gramian_bounds.npy'))
        ]):
            compute_bounds(rd, arch)

        loss = np.load(os.path.join(rd, 'loss.npy'))
        euclidean_distances = np.load(os.path.join(rd, 'euclidean_distances.npy'))
        wrm_bounds = np.load(os.path.join(rd, 'wrm_bounds.npy'))
        gramian_bounds = np.load(os.path.join(rd, 'gramian_bounds.npy'))
        lipschitz_bounds = np.load(os.path.join(rd, 'lipschitz_bounds.npy'))

        plt.plot(euclidean_distances, gramian_bounds, ls=ls, lw=2.5, color='black', zorder=10, marker='v', markevery=5)
        plt.plot(euclidean_distances, wrm_bounds, ls=ls, lw=2.0, color=colors[0], marker='d', markevery=1)
        plt.plot(euclidean_distances, lipschitz_bounds, ls=ls, lw=2.0, color=colors[1], marker='o', markevery=3)

        plt.hlines(y=loss, xmin=0.0, xmax=4.0, label=r'$\mathbb{E}_P[\ell(X,\,Y)]$', color=colors[2], zorder=11, ls=ls,
                   lw=1.5)

    # legend
    custom_lines = [Line2D([0], [0], color='black', lw=2, ls='-', marker='v'),
                    Line2D([0], [0], color=colors[0], lw=2, ls='-', marker='d'),
                    Line2D([0], [0], color=colors[1], lw=2, ls='-', marker='o'),
                    Line2D([0], [0], color=colors[2], lw=2, ls='-')]
    custom_lines = custom_lines + [Line2D([0], [0], color='gray', ls=ls, lw=2.0) for ls in linestyles]
    custom_labels = [GRAMIAN_METHOD_LEGEND, WRM_CERTIFICATE, LIPSCHITZ_CERT, r'$\mathbb{E}_P[\ell(X,\,Y)]$']
    custom_labels += progression_labels

    plt.ylabel(YLABEL)
    plt.xlabel(XLABEL)
    plt.ylim((0.0, 1.25))
    plt.xlim((-0.1, 4.0))
    plt.legend(custom_lines, custom_labels, frameon=True, fancybox=False, handletextpad=0.1, labelspacing=.2,
               handlelength=0.9, columnspacing=0.5, ncol=2, loc='lower right', bbox_to_anchor=(1.00, 0.13),
               fontsize=font_size - 1, facecolor='white', framealpha=1.0)
    fig.tight_layout()

    if save_figure_as is not None:
        plt.savefig(save_figure_as, bbox_inches='tight', pad_inches=0.1, dpi=200)
        return

    plt.show()


if __name__ == '__main__':
    figures_dir = './results/figures/comparison/'
    data_dir = './results/data/'

    ####################################################################################################################
    # vayring width
    make_plot_multiple(root_dirs=[
        data_dir + 'gaussian-mixture/progression-h=2-w=2/',
        data_dir + 'gaussian-mixture/progression-h=2-w=4/',
        data_dir + 'gaussian-mixture/progression-h=2-w=8/',
        data_dir + 'gaussian-mixture/progression-h=2-w=16/',
    ],
        linestyles=['-', '--', ':', '-.'],
        progression_labels=[r'$w=2$', r'$w=4$', r'$w=8$', r'$w=16$'],
        save_figure_as=os.path.join(figures_dir, 'width-progression.pdf')
    )

    ####################################################################################################################
    # varying depth
    make_plot_multiple(root_dirs=[
        data_dir + 'gaussian-mixture/progression-h=2-w=2/',
        data_dir + 'gaussian-mixture/progression-h=5-w=2/',
        data_dir + 'gaussian-mixture/progression-h=10-w=2/',
        data_dir + 'gaussian-mixture/progression-h=20-w=2/',
    ],
        linestyles=['-', '--', ':', '-.'],
        progression_labels=[r'$n_h=2$', r'$n_h=5$', r'$n_h=10$', r'$n_h=20$'],
        save_figure_as=os.path.join(figures_dir, 'depth-progression.pdf')
    )
