import numpy as np
import os
import random
import matplotlib.pyplot as plt

from constants import *
from lib.certify import CertifyGramian
from lib.certify.certify_utils import compute_scores
from lib.utils.plot_utils import init_style

########################################################################################################################
# plot params
FONTSIZE = 18
LINEWIDTH = 1.0
COLORS = init_style(font_size_base=FONTSIZE, linewdith_base=LINEWIDTH, sns_style='whitegrid')

########################################################################################################################
# Labels
GRAMIAN_UPPER_BOUND = 'Gramian Upper Bd.'
GRAMIAN_LOWER_BOUND = 'Gramian Lower Bd.'
UPPER_BOUND_MARKERS = 'v'
LOWER_BOUND_MARKERS = '^'

XLABEL = r'$\frac{1}{\sqrt{2}}\||\sqrt{p_{x|y}} - \sqrt{q_{x|y}}\||_2$'
YLABELS = {
    JSD_LOSS: 'JSD Loss',
    CLASSIFICATION_ERROR: 'Classification Error',
    CLASSIFICATION_ACCURACY: 'Classification Accuracy',
    AUC_SCORE: 'AUC Score'
}

EMP_SCORE_LABEL = r'$\mathbb{E}_P[\ell(h(X),\,Y)]$'

# init seed
SEED = 743
np.random.seed(SEED)
random.seed(SEED)


def compute_lines(data_file_p, data_file_q, func, save_data_as=None):
    # load data from distribution q
    test_data = np.load(data_file_q, allow_pickle=True)[()]
    test_labels = test_data['labels'].astype(int)
    test_logits = test_data['logits']

    # convert to 2dim logits
    test_logits = np.concatenate([-test_logits, test_logits], axis=1)

    # load data from distribution p
    train_data = np.load(data_file_p, allow_pickle=True)[()]
    train_labels = train_data['labels'].astype(int)
    train_logits = train_data['logits']

    # subsample such that sizes match
    n_test = len(test_labels)
    train_subset_idx = np.random.choice(np.arange(0, len(train_labels)), size=n_test, replace=False)
    train_labels = train_labels[train_subset_idx]
    train_logits = train_logits[train_subset_idx]

    # convert to 2dim logits
    train_logits = np.concatenate([-train_logits, train_logits], axis=1)

    # compute scores on P and Q
    empirical_score_train = compute_scores(train_logits, train_labels, func=func, reduce=REDUCE_MEAN)
    empirical_score_test = compute_scores(test_logits, test_labels, func=func, reduce=REDUCE_MEAN)

    print(f'Train Score: {empirical_score_train}')
    print(f'Test Score: {empirical_score_test}')

    # mix dataset
    num_samples = len(train_labels)
    indices = np.arange(0, num_samples).astype(int)
    mixing_values = np.linspace(0, 1, 20)
    mixed_scores, sampled_distances = [], []
    for gamma in mixing_values:
        # sample gamma from P
        num_from_train = int(gamma * num_samples)
        indices_train = np.random.choice(indices, num_from_train, replace=False)

        # sample 1 - gamma from Q
        indices_test = np.setdiff1d(indices, indices_train)

        # mix
        mixed_logits = np.concatenate([train_logits[indices_train, :], test_logits[indices_test, :]], axis=0)
        mixed_labels = np.concatenate([train_labels[indices_train], test_labels[indices_test]], axis=0)

        # compute score on mixed distribution
        dist = np.sqrt(1 - np.sqrt(gamma))
        mixed_score = compute_scores(mixed_logits, mixed_labels, func=func, reduce=REDUCE_MEAN)

        sampled_distances.append(dist)
        mixed_scores.append(mixed_score)

    # distances
    hellinger_distances = np.linspace(0, 1, 50)
    sampled_distances = np.array(sampled_distances)

    # adjust hellinger distances if score function is AUC
    if func == AUC_SCORE:
        sampled_distances = np.sqrt(sampled_distances ** 2 * (2 - sampled_distances ** 2))
        hellinger_distances = np.sqrt(hellinger_distances ** 2 * (2 - hellinger_distances ** 2))

    # run certification
    certify_gramian = CertifyGramian(train_logits, train_labels, func=func, finite_sampling=True)
    gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)
    gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)

    data = dict(hellinger_distances=hellinger_distances,
                sampled_distances=sampled_distances,
                gramian_upper_bounds=gramian_upper_bounds,
                gramian_lower_bounds=gramian_lower_bounds,
                empirical_score_train=empirical_score_train,
                mixed_scores=mixed_scores)

    if save_data_as:
        np.save(save_data_as, data)
        print(f'saved data as {save_data_as}')

    return data


def make_plot(hellinger_distances,
              sampled_distances,
              gramian_upper_bounds,
              gramian_lower_bounds,
              empirical_score_train,
              mixed_scores,
              func,
              upper_bound=True,
              save_figure=None):
    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    ax.scatter(sampled_distances, mixed_scores, label=r'$\mathbb{E}_{\Pi_\gamma}[\ell(X,\,Y)]$', marker='o', alpha=0.7,
               color='dimgray', s=20)
    ax.plot([0], empirical_score_train, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    ax.hlines(y=empirical_score_train, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=20, ls='--',
              lw=2.0)

    if upper_bound:
        ax.plot(hellinger_distances, gramian_upper_bounds, label=GRAMIAN_UPPER_BOUND, linestyle='-', lw=2.5,
                c='black', marker=UPPER_BOUND_MARKERS, markevery=5)
    else:
        ax.plot(hellinger_distances, gramian_lower_bounds, label=GRAMIAN_LOWER_BOUND, linestyle='-', lw=2.5, c='black',
                marker=LOWER_BOUND_MARKERS, markevery=5)

    ax.set_ylabel(YLABELS[func])
    ax.set_xlabel(XLABEL)
    ax.set_ylim((-0.05, 1.15))
    fig.tight_layout()
    plt.legend(frameon=True, loc='best', ncol=1, fontsize=FONTSIZE - 2, framealpha=1.0, fancybox=False)

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.savefig(save_figure.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_figure}')
        return

    plt.show()


def main():
    figures_dir = 'results/figures/colored-mnist'
    data_root = 'results/data/colored-mnist/'

    ####################################################################################################################
    # classification accuracy
    save_as = os.path.join(figures_dir, f'data/{CLASSIFICATION_ACCURACY}-colour-mnist-data.npy')
    cls_acc_data = compute_lines(data_file_p=data_root + 'train/train-predictions.npy',
                                 data_file_q=data_root + 'test/test-predictions.npy',
                                 func=CLASSIFICATION_ACCURACY,
                                 save_data_as=save_as)
    make_plot(**cls_acc_data, func=CLASSIFICATION_ACCURACY, save_figure=os.path.join(figures_dir, 'cmnist-accuracy.pdf'),
              upper_bound=False)

    ####################################################################################################################
    # classification error
    save_as = os.path.join(figures_dir, f'data/{CLASSIFICATION_ERROR}-colour-mnist-data.npy')
    cls_err_data = compute_lines(data_file_p=data_root + 'train/train-predictions.npy',
                                 data_file_q=data_root + 'test/test-predictions.npy',
                                 func=CLASSIFICATION_ERROR,
                                 save_data_as=save_as)
    make_plot(**cls_err_data, func=CLASSIFICATION_ERROR, save_figure=os.path.join(figures_dir, 'cmnist-error.pdf'),
              upper_bound=True)

    ####################################################################################################################
    # jsd loss
    save_as = os.path.join(figures_dir, f'data/{JSD_LOSS}-colour-mnist-data.npy')
    jsd_data = compute_lines(data_file_p=data_root + 'train/train-predictions.npy',
                             data_file_q=data_root + 'test/test-predictions.npy',
                             func=JSD_LOSS,
                             save_data_as=save_as)
    make_plot(**jsd_data, func=JSD_LOSS, save_figure=os.path.join(figures_dir, 'cmnist-jsd-loss.pdf'), upper_bound=True)

    ####################################################################################################################
    # auc score
    save_as = os.path.join(figures_dir, f'data/{AUC_SCORE}-colour-mnist-data.npy')
    auc_data = compute_lines(data_file_p=data_root + 'train/train-predictions.npy',
                             data_file_q=data_root + 'test/test-predictions.npy',
                             func=AUC_SCORE,
                             save_data_as=save_as)
    make_plot(**auc_data, func=AUC_SCORE, save_figure=os.path.join(figures_dir, 'cmnist-auc-score.pdf'),
              upper_bound=False)


if __name__ == '__main__':
    main()
