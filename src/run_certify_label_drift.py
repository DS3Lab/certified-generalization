import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from tqdm import trange

from constants import *
from lib.certify import CertifyGramian
from lib.certify.certify_utils import compute_scores
from lib.utils.plot_utils import init_style

# plot params
font_size = 16
linewidth = 1.0
colors = init_style(font_size_base=font_size, linewdith_base=linewidth)

USE_CUDA = torch.cuda.is_available()

SEED = 742

# init seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def main(data_root, loss_func, counts_source, num_samples, add_classes, max_per_new_class, save_data, do_sample=True,
         show_plot=False, class_steps=1):
    if do_sample:
        add_classes = np.arange(add_classes[0], add_classes[1] + 1, step=class_steps)
        data = sample_data(data_root=data_root,
                           loss_func=loss_func,
                           class_counts_source=counts_source,
                           num_samples=num_samples,
                           add_classes=add_classes,
                           max_per_new_class=max_per_new_class,
                           save_data=save_data)
    else:
        data = np.load(file=os.path.join(data_root, f'{loss_func}-{num_samples}-sampled-data.npy'),
                       allow_pickle=True)[()]

    # make figure
    if show_plot:
        make_plot(logits=data['source-logits'],
                  labels=data['source-labels'],
                  loss_func=loss_func,
                  empirical_source_score=data['source-score'],
                  finite_sampling=True,
                  sampled_distances=data['sampled-distances'],
                  sampled_scores=data['sampled-scores'],
                  save_figure=None)


def sample_data(data_root, loss_func, class_counts_source, num_samples=100, add_classes: np.ndarray = np.array([]),
                max_per_new_class: int = 0, save_data=True):
    # load data (contains entire dataset)
    data = np.load(os.path.join(data_root, 'test-predictions.npy'), allow_pickle=True)[()]
    labels = data['labels'].astype(int)
    logits = data['logits']
    _, class_counts = np.unique(labels, return_counts=True)

    if class_counts_source is None:
        class_counts_source = class_counts

    # subsample entire dataset so that it matches P0
    num_classes = np.max(labels) + 1
    source_labels, source_logits = adjust_source_distribution(labels, logits, class_counts, class_counts_source,
                                                              num_classes)

    # sample points
    sampled_distances, sampled_scores, source_score = sample_score_points(logits=logits,
                                                                          labels=labels,
                                                                          source_logits=source_logits,
                                                                          source_labels=source_labels,
                                                                          num_classes=num_classes,
                                                                          loss_func=loss_func,
                                                                          class_counts=class_counts,
                                                                          num_samples=num_samples,
                                                                          add_classes=add_classes,
                                                                          max_per_new_class=max_per_new_class)

    # save everything
    data = {'source-logits': source_logits,
            'source-labels': source_labels,
            'source-score': source_score,
            'sampled-distances': sampled_distances,
            'sampled-scores': sampled_scores}

    if save_data:
        np.save(file=os.path.join(data_root, f'{loss_func}-{num_samples}-sampled-data.npy'), arr=data)

    return data


def sample_score_points(logits,
                        labels,
                        source_logits,
                        source_labels,
                        num_classes,
                        loss_func,
                        class_counts,
                        num_samples=100,
                        add_classes: np.ndarray = np.array([]),
                        max_per_new_class: int = 0):
    _, class_counts_source = np.unique(source_labels, return_counts=True)
    source_distribution = class_counts_source / np.sum(class_counts_source)

    # compute loss on source distribution
    source_score = compute_scores(source_logits, source_labels, func=loss_func, reduce=REDUCE_MEAN)
    num_samples = num_samples // len(add_classes)
    sampled_distances, sampled_scores = [], []

    for ac in add_classes:
        dists, losses = sample_q_distribution(ac, source_distribution, labels, logits, class_counts,
                                              num_classes, loss_func, max_per_new_class, num_samples)
        sampled_distances.append(dists)
        sampled_scores.append(losses)

    return sampled_distances, sampled_scores, source_score


def sample_q_distribution(add_classes, source_distribution, labels, logits, class_counts, num_classes,
                          loss_func, max_per_new_class, num_samples):
    # in case we include classes, expand the probability vector of the source domain
    if add_classes > 0:
        source_distribution = np.concatenate([source_distribution, np.zeros(shape=add_classes)], axis=0)

    # subsample
    sampled_distances, sampled_scores = [], []
    progress_bar = trange(num_samples, leave=True)
    for i, _ in enumerate(progress_bar):
        # sample p1 and compute loss
        empirical_score_target, target_distribution = generate_shifted_label_distribution(
            labels, logits, class_counts, num_classes, loss_func, add_classes, max_per_new_class)

        # compute hellinger distance
        dist = np.sqrt(0.5 * np.sum((np.sqrt(source_distribution) - np.sqrt(target_distribution)) ** 2))

        sampled_distances.append(dist)
        sampled_scores.append(empirical_score_target)

        progress_bar.set_description(f'generated {i + 1}/{num_samples} samples for add-classes={add_classes}')
        progress_bar.refresh()

    return sampled_distances, sampled_scores


def generate_shifted_label_distribution(labels, logits, class_counts, num_classes, loss_func, add_classes: int = 0,
                                        max_per_new_class: int = 1):
    # generate P1
    shifted_class_counts = [np.random.randint(0, n, 1)[0] for n in class_counts]

    # add new classes (i.e. those which the classifier has not seen)
    class_counts_new = []
    if add_classes > 0 and max_per_new_class > 0:
        class_counts_new = [np.random.randint(0, max_per_new_class, 1)[0] for _ in range(add_classes)]
    elif add_classes < 0:
        # randomly drop add_classes classes
        drop_classes = np.random.choice(np.arange(0, num_classes), -add_classes, replace=False)
        shifted_class_counts = [0 if i in drop_classes else s for i, s in enumerate(shifted_class_counts)]

    num_new_samples = np.sum(class_counts_new)
    distribution1 = (shifted_class_counts + class_counts_new) / np.sum(shifted_class_counts + class_counts_new)

    # subsample
    label_indices = np.array([np.squeeze(np.nonzero(labels == k)) for k in range(num_classes)])
    label_indices_subsampled = [np.random.choice(idx, n) for idx, n in zip(label_indices, shifted_class_counts)]
    label_indices_subsampled = np.concatenate(label_indices_subsampled)
    subsampled_labels = labels[label_indices_subsampled]
    subsampled_logits = logits[label_indices_subsampled]

    # generate logits for new classes (=-np.inf)
    new_num_classes = num_classes
    if add_classes > 0 and max_per_new_class > 0:
        new_num_classes += add_classes
        new_logits = np.ones(shape=(len(subsampled_labels), add_classes)) * (-np.inf)
        subsampled_logits = np.concatenate([subsampled_logits, new_logits], axis=1)

        # generate logits + labels for new samples
        new_logits = np.zeros(shape=(num_new_samples, num_classes + add_classes))
        new_logits[:, 0] = 1.0
        subsampled_logits = np.concatenate([subsampled_logits, new_logits], axis=0)
        new_labels = np.ones(shape=num_new_samples, dtype=int) * (new_num_classes - 1)
        subsampled_labels = np.concatenate([subsampled_labels, new_labels], axis=0)

    # compute loss on shifted label distribution
    empirical_loss = compute_scores(subsampled_logits, subsampled_labels, func=loss_func,
                                    num_classes=new_num_classes, reduce=REDUCE_MEAN)

    return empirical_loss, distribution1


def adjust_source_distribution(all_labels, all_logits, all_class_counts, target_class_counts, num_classes):
    target_class_counts = np.array(target_class_counts, dtype=int)
    if not np.array(all_class_counts >= target_class_counts).all():
        # adjust class counts
        pivot_ratio = np.max(target_class_counts.astype(float) / all_class_counts.astype(float)) ** (-1)
        target_class_counts = np.ceil(float(pivot_ratio) * target_class_counts).astype(int)

    # subsample
    indices_per_class = np.array([np.squeeze(np.nonzero(all_labels == k)) for k in range(num_classes)])
    label_indices_subsampled = [np.random.choice(idx, n) for idx, n in zip(indices_per_class, target_class_counts)]
    label_indices_subsampled = np.concatenate(label_indices_subsampled)
    subsampled_labels = all_labels[label_indices_subsampled]
    subsampled_logits = all_logits[label_indices_subsampled]

    return subsampled_labels, subsampled_logits


def make_plot(logits, labels, loss_func, empirical_source_score, finite_sampling, sampled_distances, sampled_scores,
              save_figure):
    # distances
    hellinger_distances = np.linspace(0, 1, 50)

    # run certification
    certify_gramian = CertifyGramian(logits, labels, func=loss_func, finite_sampling=finite_sampling)
    gramian_lower_bounds = certify_gramian.certify(hellinger_distances, upper_bound=False)
    gramian_upper_bounds = certify_gramian.certify(hellinger_distances, upper_bound=True)

    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(hellinger_distances, gramian_lower_bounds, label=r'Gramian', linestyle='-', lw=1.0, c=colors[1])
    ax.plot(hellinger_distances, gramian_upper_bounds, linestyle='-', lw=1.0, c=colors[1])
    ax.scatter([0], empirical_source_score, label=r'$\mathbb{E}_P[\ell(X,\,Y)]$', marker='x', color=colors[2], s=50)
    ax.scatter(sampled_distances, sampled_scores, marker='o', alpha=0.7, color='dimgray', s=2,
               label=r'$\mathbb{E}_Q[\ell(X,\,Y)]$')
    ax.set_ylabel('JSD-Loss' if loss_func == JSD_LOSS else 'Classification Error')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylim((0.0, 1.05))
    fig.tight_layout()
    plt.legend(frameon=True, loc='best', ncol=1, fontsize=font_size - 4)

    if save_figure is not None:
        plt.savefig(save_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        return

    plt.show()


if __name__ == '__main__':
    # Cifar
    CIFAR10_BALANCED_SPECS = dict(loss_func=JSD_LOSS,
                                  counts_source=None,
                                  do_sample=True,
                                  data_root='results/data/cifar10/cifar-resnet110',
                                  num_samples=100000,
                                  save_data=True,
                                  add_classes=[-8, 20],
                                  max_per_new_class=2000)

    # YELP (5-class)
    YELP_SPECS = dict(loss_func=JSD_LOSS,
                      counts_source=None,
                      do_sample=True,
                      data_root='results/data/yelp/BERT_logits',
                      num_samples=100000,
                      save_data=True,
                      add_classes=[-3, 10],
                      max_per_new_class=500)

    # SNLI
    SNLI_SPECS = dict(loss_func=JSD_LOSS,
                      counts_source=None,
                      do_sample=True,
                      data_root='results/data/SNLI/DeBERTa_logits',
                      num_samples=100000,
                      save_data=True,
                      add_classes=[-1, 10],
                      max_per_new_class=3000)

    # Imagenet
    IMAGENET_SPECS_JSD = dict(loss_func=JSD_LOSS,
                              counts_source=None,
                              do_sample=True,
                              data_root='results/data/imagenet/efficientnet_b7',
                              num_samples=10000,
                              save_data=True,
                              add_classes=[-500, 500],
                              class_steps=25,
                              max_per_new_class=1000)

    IMAGENET_SPECS_ERR = dict(loss_func=CLASSIFICATION_ERROR,
                              counts_source=None,
                              do_sample=True,
                              data_root='results/data/imagenet/efficientnet_b7',
                              num_samples=10000,
                              save_data=True,
                              add_classes=[-500, 500],
                              class_steps=25,
                              max_per_new_class=1000)

    # main(**IMAGENET_SPECS_JSD, show_plot=False)
    main(**IMAGENET_SPECS_ERR, show_plot=False)
    # main(**YELP_SPECS, show_plot=False)
    # main(**CIFAR10_BALANCED_SPECS, show_plot=False)
