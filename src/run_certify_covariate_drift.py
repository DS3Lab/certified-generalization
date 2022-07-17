import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from constants import *
from lib.certify import CertifyGramian
from lib.certify.certify_utils import compute_scores
from lib.utils.plot_utils import init_style

# plot params
font_size = 16
linewidth = 1.0
colors = init_style(font_size_base=font_size, linewdith_base=linewidth)

GRAMIAN_METHOD_LEGEND = 'Gramian Certificate'
XLABEL = r'Hellinger Distance $\rho$'

YLABELS = {
    JSD_LOSS: 'Jensen-Shannon Divergence Loss',
    CLASSIFICATION_ERROR: 'Classification Error',
    CLASSIFICATION_ACCURACY: 'Classification Accuracy',
    AUC_SCORE: 'AUC Score'
}

SEED = 742

# init seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def run_certify_covariate_drift_old(data_file, data_file_q, func, save_figure=None, upper_bound=True):
    # load data from distribution p
    data_p = np.load(data_file, allow_pickle=True)[()]
    labels_p = data_p['labels'].astype(int)
    logits_p = data_p['logits']

    # load data from distribution q
    data_q = np.load(data_file_q, allow_pickle=True)[()]
    labels_q = data_q['labels'].astype(int)
    logits_q = data_q['logits']

    # compute scores on P and Q
    empirical_score_p = compute_scores(logits_p, labels_p, func=func, reduce=REDUCE_MEAN)

    # mix dataset
    num_samples = len(labels_p)
    indices = np.arange(0, num_samples).astype(int)
    mixing_values = np.linspace(0, 1, 20)
    mixed_scores, sampled_distances = [], []
    for gamma in mixing_values:
        # sample gamma from P
        num_from_p = int(gamma * num_samples)
        indices_p = np.random.choice(indices, num_from_p, replace=False)

        # sample 1 - gamma from Q
        indices_q = np.setdiff1d(indices, indices_p)

        # mix
        mixed_logits = np.concatenate([logits_p[indices_p, :], logits_q[indices_q, :]], axis=0)
        mixed_labels = np.concatenate([labels_p[indices_p], labels_q[indices_q]], axis=0)

        # compute score on mixed distribution
        dist = np.sqrt(1 - np.sqrt(gamma))
        mixed_score = compute_scores(mixed_logits, mixed_labels, func=func, reduce=REDUCE_MEAN)

        sampled_distances.append(dist)
        mixed_scores.append(mixed_score)

    # distances
    hellinger_distances = np.linspace(0, 1, 50)

    # run certification
    certify_gramian = CertifyGramian(logits_p, labels_p, func=func, finite_sampling=True)
    gramian_bounds = certify_gramian.certify(hellinger_distances, upper_bound=upper_bound)

    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(hellinger_distances, gramian_bounds, label=GRAMIAN_METHOD_LEGEND, linestyle='-', lw=1.5, c='black')
    ax.scatter([0], empirical_score_p, label=r'$\mathbb{E}_P[\ell(X,\,Y)]$', marker='x', color=colors[1], s=50)
    ax.scatter(sampled_distances, mixed_scores, label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$', marker='o', alpha=0.7,
               color='dimgray', s=20)
    ax.set_ylabel(YLABELS[func])
    ax.set_xlabel(XLABEL)
    ax.set_ylim((-0.05, 1.15))
    fig.tight_layout()
    plt.legend(frameon=True, loc='best', ncol=2, fontsize=font_size - 4)

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.savefig(save_figure.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_figure}')
        return

    plt.show()


def run_certify_covariate_drift(data_file_p, data_file_q, func, save_figure=None, upper_bound=True):
    # load data from distribution p
    data_p = np.load(data_file_p, allow_pickle=True)[()]
    labels_p = data_p['labels'].astype(int)
    logits_p = data_p['logits']

    # load data from distribution q
    data_q = np.load(data_file_q, allow_pickle=True)[()]
    labels_q = data_q['labels'].astype(int)
    logits_q = data_q['logits']

    # compute scores on P and Q
    empirical_score_p = compute_scores(logits_p, labels_p, func=func, reduce=REDUCE_MEAN)

    # mix dataset
    num_samples = len(labels_p)
    indices = np.arange(0, num_samples).astype(int)
    mixing_values = np.linspace(0, 1, 20)
    mixed_scores, sampled_distances = [], []
    for gamma in mixing_values:
        # sample gamma from P
        num_from_p = int(gamma * num_samples)
        indices_p = np.random.choice(indices, num_from_p, replace=False)

        # sample 1 - gamma from Q
        indices_q = np.setdiff1d(indices, indices_p)

        # mix
        mixed_logits = np.concatenate([logits_p[indices_p, :], logits_q[indices_q, :]], axis=0)
        mixed_labels = np.concatenate([labels_p[indices_p], labels_q[indices_q]], axis=0)

        # compute score on mixed distribution
        dist = np.sqrt(1 - np.sqrt(gamma))
        mixed_score = compute_scores(mixed_logits, mixed_labels, func=func, reduce=REDUCE_MEAN)

        sampled_distances.append(dist)
        mixed_scores.append(mixed_score)

    # distances
    hellinger_distances = np.linspace(0, 1, 50)

    # run certification
    certify_gramian = CertifyGramian(logits_p, labels_p, func=func, finite_sampling=True)
    gramian_bounds = certify_gramian.certify(hellinger_distances, upper_bound=upper_bound)

    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(hellinger_distances, gramian_bounds, label=GRAMIAN_METHOD_LEGEND, linestyle='-', lw=1.5, c='black')
    ax.scatter([0], empirical_score_p, label=r'$\mathbb{E}_P[\ell(X,\,Y)]$', marker='x', color=colors[1], s=50)
    ax.scatter(sampled_distances, mixed_scores, label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$', marker='o', alpha=0.7,
               color='dimgray', s=20)
    ax.set_ylabel(YLABELS[func])
    ax.set_xlabel(XLABEL)
    ax.set_ylim((-0.05, 1.15))
    fig.tight_layout()
    plt.legend(frameon=True, loc='best', ncol=2, fontsize=font_size - 4)

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.savefig(save_figure.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_figure}')
        return

    plt.show()


if __name__ == '__main__':
    save_as = f'./results/figures/covariate-drift/{CLASSIFICATION_ERROR}-imagenet-effnetb7-covariate-drift.pdf'
    run_certify_covariate_drift(
        data_file_p='results/data/imagenet/efficientnet_b7/test-predictions.npy',
        data_file_q='results/data/imagenet-grayscale/efficientnet_b7/test-predictions.npy',
        func=CLASSIFICATION_ERROR, upper_bound=True, save_figure=None)

    # save_as = f'./results/figures/covariate-drift/{JSD_LOSS}-imagenet-effnetb7-covariate-drift.pdf'
    # run_certify_covariate_drift(
    #     data_file_p='results/data/imagenet/efficientnet_b7/test-predictions.npy',
    #     data_file_q='results/data/imagenet-grayscale/efficientnet_b7/test-predictions.npy',
    #     func=JSD_LOSS, upper_bound=True, save_figure=save_as)
