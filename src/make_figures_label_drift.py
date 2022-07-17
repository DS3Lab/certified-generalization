import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constants import *
from lib.certify import CertifyGramian
from lib.utils.plot_utils import init_style

########################################################################################################################
# formatting
FONTSIZE = 18
LINEWIDTH = 1.0
COLORS = init_style(font_size_base=FONTSIZE, linewdith_base=LINEWIDTH, sns_style='whitegrid')

########################################################################################################################
# Labels
GRAMIAN_UPPER = 'Gramian Upper Bd.'
GRAMIAN_LOWER = 'Gramian Lower Bd.'
UPPER_BOUND_MARKERS = 'v'
LOWER_BOUND_MARKERS = '^'

XLABEL = r'$\frac{1}{\sqrt{2}}\||\sqrt{p_y} - \sqrt{q_y}\||_2$'
YLABELS = {
    JSD_LOSS: 'JSD Loss',
    CLASSIFICATION_ERROR: 'Classification Error',
    CLASSIFICATION_ACCURACY: 'Classification Accuracy',
    AUC_SCORE: 'AUC Score'
}

EMP_SCORE_LABEL = r'$\mathbb{E}_P[\ell(h(X),\,Y)]$'
EMP_ADV_LABEL = r'$\mathbb{E}_Q[\ell(h(X),\,Y)]$'


def main_with_subplots(cifar_root, yelp_root, save_as):
    # distances
    hellinger_distances = np.linspace(0, 1, 50)

    ####################################################################################################################
    # cifar
    cifar_data = np.load(file=os.path.join(cifar_root, 'sampled-error-jsd-data.npy'), allow_pickle=True)[()]
    cifar_logits = cifar_data['source-logits']
    cifar_labels = cifar_data['source-labels']
    cifar_empirical_source_score = cifar_data['source-classification-error']
    cifar_sampled_scores = cifar_data['sampled-classification-errors']
    cifar_sampled_distances = cifar_data['sampled-distances']

    # run certification
    cifar_certify_gramian = CertifyGramian(cifar_logits, cifar_labels, func=CLASSIFICATION_ERROR, finite_sampling=True)
    cifar_gramian_lower_bounds = cifar_certify_gramian.certify(hellinger_distances, upper_bound=False)
    cifar_gramian_upper_bounds = cifar_certify_gramian.certify(hellinger_distances, upper_bound=True)

    ####################################################################################################################
    # yelp
    yelp_data = np.load(file=os.path.join(yelp_root, 'sampled-error-jsd-data.npy'), allow_pickle=True)[()]
    yelp_logits = yelp_data['source-logits']
    yelp_labels = yelp_data['source-labels']
    yelp_empirical_source_score = yelp_data['source-classification-error']
    yelp_sampled_scores = yelp_data['sampled-classification-errors']
    yelp_sampled_distances = yelp_data['sampled-distances']

    # run certification
    yelp_certify_gramian = CertifyGramian(yelp_logits, yelp_labels, func=CLASSIFICATION_ERROR, finite_sampling=True)
    yelp_gramian_lower_bounds = yelp_certify_gramian.certify(hellinger_distances, upper_bound=False)
    yelp_gramian_upper_bounds = yelp_certify_gramian.certify(hellinger_distances, upper_bound=True)

    ####################################################################################################################
    # make figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # make cifar subfig
    ax = axs[0]
    ax.plot(hellinger_distances, cifar_gramian_lower_bounds, label=GRAMIAN_LOWER, linestyle='-', lw=2.5, c='black',
            marker=LOWER_BOUND_MARKERS, markevery=2)
    ax.plot(hellinger_distances, cifar_gramian_upper_bounds, label=GRAMIAN_UPPER, linestyle='-', lw=2.5, c='black',
            marker=UPPER_BOUND_MARKERS, markevery=2)
    ax.scatter(cifar_sampled_distances, cifar_sampled_scores, marker='o', alpha=0.7, color='dimgray', s=2,
               label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$')

    ax.plot([0], cifar_empirical_source_score, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    ax.hlines(y=cifar_empirical_source_score, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=20,
              ls='--', lw=2.0)
    # ax.scatter([0], cifar_empirical_source_score, label=EMP_SCORE_LABEL, marker='x', color=COLORS[1], s=50, zorder=20)

    ax.set_ylabel(YLABELS[CLASSIFICATION_ERROR])
    ax.set_xlabel(XLABEL)
    ax.set_ylim((-0.02, 0.6))
    ax.set_xlim((-0.02, 0.3))
    ax.set_title('CIFAR-10')

    # make yelp subfig
    ax = axs[1]
    ax.plot(hellinger_distances, yelp_gramian_lower_bounds, linestyle='-', lw=2.5, c='black',
            marker=LOWER_BOUND_MARKERS, markevery=2)
    ax.plot(hellinger_distances, yelp_gramian_upper_bounds, linestyle='-', lw=2.5, c='black',
            marker=UPPER_BOUND_MARKERS, markevery=2)

    ax.scatter(yelp_sampled_distances, yelp_sampled_scores, marker='o', alpha=0.7, color='dimgray', s=2,
               label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$')

    ax.plot([0], yelp_empirical_source_score, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    ax.hlines(y=yelp_empirical_source_score, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=20,
              ls='--', lw=2.0)

    ax.set_xlabel(XLABEL)
    ax.set_ylim((-0.02, 1.0))
    ax.set_xlim((-0.02, 0.5))
    ax.set_title('Yelp')

    # legend
    handles = [
        # Line2D([0], [0], color='black', lw=2.5, ls='-'),
        Line2D([0], [0], color=COLORS[2], marker='D', ls='-', lw=2.0),
        Line2D([0], [0], color=COLORS[1], marker='o', alpha=0.7, ls='', c='dimgray', lw=2.0),
        Line2D([0], [0], color='black', marker=UPPER_BOUND_MARKERS, ls='-', lw=2.0),
        Line2D([0], [0], color='black', marker=LOWER_BOUND_MARKERS, ls='-', lw=2.0)]
    labels = [EMP_SCORE_LABEL, EMP_ADV_LABEL, GRAMIAN_UPPER, GRAMIAN_LOWER]
    fig.legend(handles, labels, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               fontsize=FONTSIZE - 2, handletextpad=0.5, labelspacing=1.0, handlelength=1.5, columnspacing=1.0)

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.savefig(save_as.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)


def main_single_plot(data_root, title, func=CLASSIFICATION_ERROR, save_as=None, xlim=None, ylim=None):
    data_keys = {
        CLASSIFICATION_ERROR: {
            'empirical_source_score': 'source-classification-error',
            'sampled_scores': 'sampled-classification-errors'},
        JSD_LOSS: {
            'empirical_source_score': 'source-jsd-loss',
            'sampled_scores': 'sampled-jsd-losses'
        },
    }[func]

    # distances
    hellinger_distances = np.linspace(0, 1, 50)

    data = np.load(file=os.path.join(data_root, 'sampled-error-jsd-data.npy'), allow_pickle=True)[()]
    logits = data['source-logits']
    labels = data['source-labels']
    empirical_source_score = data[data_keys['empirical_source_score']]
    sampled_scores = data[data_keys['sampled_scores']]
    sampled_distances = data['sampled-distances']

    ####################################################################################################################
    # run certification
    certify_gramian = CertifyGramian(logits, labels, func=func, finite_sampling=True)
    gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)
    gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)

    ####################################################################################################################
    # make figure
    fig = plt.figure(figsize=(6, 4))

    # make cifar subfig
    plt.plot(hellinger_distances, gramian_lower_bounds, linestyle='-', lw=2.5, c='black',
             marker=LOWER_BOUND_MARKERS, markevery=2)
    plt.plot(hellinger_distances, gramian_upper_bounds, linestyle='-', lw=2.5, c='black',
             marker=UPPER_BOUND_MARKERS, markevery=2)

    plt.scatter(sampled_distances, sampled_scores, marker='o', alpha=0.7, color='dimgray', s=2,
                label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$')

    plt.plot([0], empirical_source_score, marker='D', color=COLORS[2], lw=2.0, zorder=20)
    plt.hlines(y=empirical_source_score, xmin=0.0, xmax=1.0, label=EMP_SCORE_LABEL, color=COLORS[2], zorder=20,
               ls='--', lw=2.0)

    plt.ylabel(YLABELS[func])
    plt.xlabel(XLABEL)
    plt.ylim(ylim or (-0.02, 0.6))
    plt.xlim(xlim or (-0.02, 0.3))
    if title is not None:
        plt.title(title)

    # legend
    handles = [Line2D([0], [0], color='black', lw=2.5, ls='-', marker=LOWER_BOUND_MARKERS),
               Line2D([0], [0], color='black', lw=2.5, ls='-', marker=UPPER_BOUND_MARKERS),
               Line2D([0], [0], color=COLORS[2], marker='D', ls='--', lw=2.0),
               Line2D([0], [0], color=COLORS[1], marker='o', alpha=0.7, ls='', c='dimgray', lw=2.0)]
    labels = [GRAMIAN_LOWER, GRAMIAN_UPPER, EMP_SCORE_LABEL, EMP_ADV_LABEL]
    plt.legend(handles, labels, frameon=True, fancybox=False, framealpha=1.0, handletextpad=0.5, labelspacing=.2,
               handlelength=1.2, columnspacing=0.8, ncol=1, loc='upper left', bbox_to_anchor=(0.0, 1.0),
               fontsize=FONTSIZE - 4)

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.savefig(save_as.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    figures_dir = './results/figures/label-drift/'
    data_dir = 'results/data'

    # ####################################################################################################################
    # # main figure with two suplots
    main_with_subplots(
        save_as=os.path.join(figures_dir, 'cifar10-yelp-label_drift.pdf'),
        cifar_root=os.path.join(data_dir, 'cifar10/densenet121'),
        yelp_root=os.path.join(data_dir, 'yelp/BERT_logits')
    )

    # ##################################################################################################################
    # individual cifar-10 figures
    for fn in [JSD_LOSS, CLASSIFICATION_ERROR]:
        save_as_fp = os.path.join(figures_dir, 'cifar10-label-drift-{}-{}.pdf')
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/densenet121'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('densenet121', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/densenet169'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('densenet169', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/googlenet'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('googlenet', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/inception_v3'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('inception_v3', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/mobilenet_v2'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('mobilenet_v2', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/resnet18'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('resnet18', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/resnet50'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('resnet50', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/vgg11_bn'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('vgg11bn', fn))
        main_single_plot(data_root=os.path.join(data_dir, 'cifar10/vgg19_bn'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('vgg19bn', fn))

    # ##################################################################################################################
    # individual imagenet figures
    for fn in [JSD_LOSS, CLASSIFICATION_ERROR]:
        save_as_fp = os.path.join(figures_dir, 'imagenet-label-drift-{}-{}.pdf')
        main_single_plot(data_root=os.path.join(data_dir, 'imagenet/efficientnet_b7'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('efficientnet_b7', fn),
                         xlim=(-0.02, 0.5), ylim=(-0.02, 1.0))

    # ##################################################################################################################
    # individual snli figures
    for fn in [JSD_LOSS, CLASSIFICATION_ERROR]:
        save_as_fp = os.path.join(figures_dir, 'snli-label-drift-{}-{}.pdf')
        main_single_plot(data_root=os.path.join(data_dir, 'snli/DeBERTa_logits'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('DeBERTa', fn),
                         xlim=(-0.02, 0.5), ylim=(-0.02, 1.0))

    # ##################################################################################################################
    # individual yelp figures
    for fn in [JSD_LOSS]:
        save_as_fp = os.path.join(figures_dir, 'yelp-label-drift-{}-{}.pdf')
        main_single_plot(data_root=os.path.join(data_dir, 'yelp/BERT_logits'),
                         title=None,
                         func=fn,
                         save_as=save_as_fp.format('BERT', fn),
                         xlim=(-0.02, 0.5), ylim=(-0.02, 1.0))
