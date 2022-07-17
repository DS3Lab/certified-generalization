import numpy as np
import os
import random
from scipy.special import softmax, rel_entr
from typing import List
from tqdm import tqdm

from constants import *
from lib.certify.certify_utils import numpy_one_hot_encode

# init seed
SEED = 742
np.random.seed(SEED)
random.seed(SEED)


def do_sample(data_root,
              n_samples,
              num_zero_classes: List[int],
              num_add_classes: List[int],
              max_samples_per_new_class: int,
              frac_change_small_dist: List[float],
              max_remove_small_dist: List[int],
              save_data: bool = True):
    # load data (contains entire dataset)
    data = np.load(os.path.join(data_root, 'test-predictions.npy'), allow_pickle=True)[()]
    labels = data['labels'].astype(int)
    logits = data['logits']
    num_classes = np.max(labels) + 1
    predicted_labels = np.argmax(logits, axis=1)
    predicted_probs = softmax(logits, axis=1)
    one_hot_labels = numpy_one_hot_encode(labels, num_classes=num_classes)

    # compute loss terms for jsd loss per sample
    jsd_loss_terms = compute_jsd_losses_per_sample(predicted_probs, one_hot_labels)
    jsd_loss = np.mean(jsd_loss_terms)

    # compute loss terms for classification error
    classification_error_terms = np.array(predicted_labels != labels).astype(float)
    classification_error = np.mean(classification_error_terms)

    # get class counts, labels, distribution 0
    classes, class_counts = np.unique(labels, return_counts=True)
    label_indices = np.array([np.squeeze(np.nonzero(labels == k)) for k in classes])
    distribution_p = class_counts / np.sum(class_counts, keepdims=True)

    # init random number generator
    rng = np.random.default_rng()

    sampled_hellinger_distances = np.empty(shape=0)
    sampled_classification_errors = np.empty(shape=0)
    sampled_jsd_losses = np.empty(shape=0)

    # 1) generate samples by changing existing class distribution
    hellinger_distances_1, classification_errors_1, jsd_losses_1 = subsample_existing_classes(
        n_samples // 4, class_counts, label_indices, distribution_p, classification_error_terms, jsd_loss_terms, rng,
        num_classes, 0)

    sampled_hellinger_distances = np.concatenate([sampled_hellinger_distances, hellinger_distances_1], axis=0)
    sampled_classification_errors = np.concatenate([sampled_classification_errors, classification_errors_1], axis=0)
    sampled_jsd_losses = np.concatenate([sampled_jsd_losses, jsd_losses_1], axis=0)

    # 2) generate random class counts  while removing classes
    n_single_remove_classes = n_samples // (4 * len(num_zero_classes))
    for nz in tqdm(num_zero_classes):
        hellinger_distances_2, classification_errors_2, jsd_losses_2 = subsample_existing_classes(
            n_single_remove_classes, class_counts, label_indices, distribution_p, classification_error_terms,
            jsd_loss_terms, rng, num_classes, nz)

        sampled_hellinger_distances = np.concatenate([sampled_hellinger_distances, hellinger_distances_2], axis=0)
        sampled_classification_errors = np.concatenate([sampled_classification_errors, classification_errors_2], axis=0)
        sampled_jsd_losses = np.concatenate([sampled_jsd_losses, jsd_losses_2], axis=0)

    # 3) generate random class counts while adding new classes
    n_single_add_classes = n_samples // (4 * len(num_add_classes))
    for na in num_add_classes:
        hellinger_distances_3, classification_errors_3, jsd_losses_3 = subsample_with_new_classes(
            n_single_add_classes, labels, logits, rng, num_classes, na, max_samples_per_new_class)

        sampled_hellinger_distances = np.concatenate([sampled_hellinger_distances, hellinger_distances_3], axis=0)
        sampled_classification_errors = np.concatenate([sampled_classification_errors, classification_errors_3], axis=0)
        sampled_jsd_losses = np.concatenate([sampled_jsd_losses, jsd_losses_3], axis=0)

    # 4) generate class counts with small hellinger distance
    n_single_samples = n_samples // (4 * len(frac_change_small_dist) * len(max_remove_small_dist))
    for frac_change in frac_change_small_dist:
        for max_remove in max_remove_small_dist:
            hellinger_distances_4, classification_errors_4, jsd_losses_4 = generate_small_hellinger_dists(
                n_single_samples, class_counts, label_indices, distribution_p, classification_error_terms,
                jsd_loss_terms, rng, frac_change, max_remove)

            sampled_hellinger_distances = np.concatenate([sampled_hellinger_distances, hellinger_distances_4], axis=0)
            sampled_classification_errors = np.concatenate([sampled_classification_errors, classification_errors_4],
                                                           axis=0)
            sampled_jsd_losses = np.concatenate([sampled_jsd_losses, jsd_losses_4], axis=0)

    # dump everything
    data = {
        'source-logits': logits,
        'source-labels': labels,
        'source-classification-error': classification_error,
        'source-jsd-loss': jsd_loss,
        'sampled-distances': sampled_hellinger_distances,
        'sampled-classification-errors': sampled_classification_errors,
        'sampled-jsd-losses': sampled_jsd_losses
    }

    if save_data:
        save_fn = os.path.join(data_root, f'sampled-error-jsd-data.npy')
        np.save(file=save_fn, arr=data)
        print(f'\n\nsaved {save_fn}\n\n')

    return data


def generate_small_hellinger_dists(n_samples, class_counts, label_indices, distribution_p, classification_error_terms,
                                   jsd_loss_terms, rng, frac_change, max_change):
    sampled_class_counts = np.squeeze(np.array([
        np.tile([nk], (n_samples, 1)) -
        np.random.binomial(1, frac_change, n_samples).reshape(n_samples, 1)
        * np.random.randint(0, max_change, n_samples).reshape(n_samples, 1)
        for nk in class_counts
    ])).T

    # compute hellinger distances
    sampled_distributions_q = sampled_class_counts / np.sum(sampled_class_counts, axis=1,
                                                            keepdims=True)
    distribution_p_tiled = np.tile(distribution_p, (n_samples, 1))
    squares = (np.sqrt(sampled_distributions_q) - np.sqrt(distribution_p_tiled)) ** 2
    hellinger_distances = np.sqrt(0.5 * np.sum(squares, axis=1))

    # sample indices for each class count vector
    classiciation_errors, jsd_losses = [], []
    for random_class_counts in tqdm(sampled_class_counts):
        indices = np.concatenate([
            rng.permutation(class_indices)[:n_k] for n_k, class_indices in zip(random_class_counts, label_indices)
        ], axis=0)

        # compute classification error loss
        classiciation_errors.append(np.mean(classification_error_terms[indices]))

        # compute jsd loss
        jsd_losses.append(np.mean(jsd_loss_terms[indices]))

    return hellinger_distances, classiciation_errors, jsd_losses


def subsample_with_new_classes(n_samples, labels, logits, rng, num_classes, num_add_classes, max_samples_per_new_class):
    # adjust initial distribution to account for new classes
    _, class_counts0 = np.unique(labels, return_counts=True)
    class_counts0 = np.concatenate([class_counts0, np.zeros(shape=num_add_classes)])
    distribution_p = class_counts0 / np.sum(class_counts0, keepdims=True)

    # include samples for new classes
    new_num_classes = num_classes + num_add_classes
    new_logits = np.zeros(shape=(max_samples_per_new_class * num_add_classes, num_classes + num_add_classes))
    new_logits[:, 0] = 1
    new_labels = np.concatenate(
        [np.ones(shape=max_samples_per_new_class) * (num_classes + i) for i in range(num_add_classes)])

    # merge with existing logits / labels
    num_samples_0 = len(labels)
    logits = np.concatenate([logits, -np.inf * np.ones(shape=(num_samples_0, num_add_classes))], axis=1)
    logits = np.concatenate([logits, new_logits], axis=0).astype(float)
    labels = np.concatenate([labels, new_labels], axis=0).astype(int)
    predicted_labels = np.argmax(logits, axis=1)
    predicted_probs = softmax(logits, axis=1)
    one_hot_labels = numpy_one_hot_encode(labels, num_classes=num_classes + num_add_classes)

    # compute loss terms for jsd loss per sample
    jsd_loss_terms = compute_jsd_losses_per_sample(predicted_probs, one_hot_labels)

    # compute loss terms for classification error
    classification_error_terms = np.array(predicted_labels != labels).astype(float)

    # get class counts, labels, distribution 0
    classes, class_counts = np.unique(labels, return_counts=True)
    label_indices = np.array([np.squeeze(np.nonzero(labels == k)) for k in classes])

    # free up memory
    del logits
    del labels

    hellinger_distances, classification_errors, jsd_losses = subsample_existing_classes(
        n_samples, class_counts, label_indices, distribution_p, classification_error_terms, jsd_loss_terms, rng,
        new_num_classes, num_zero_classes=0, new_class_counts_min=None)

    for n_remove in np.arange(1, 10):
        new_class_counts_min = np.array(
            [max(0, nk - n_remove) if i < num_classes else 0 for i, nk in enumerate(class_counts)])
        h, c, jsd = subsample_existing_classes(
            n_samples, class_counts, label_indices, distribution_p, classification_error_terms, jsd_loss_terms, rng,
            new_num_classes, num_zero_classes=0, new_class_counts_min=new_class_counts_min)

        hellinger_distances = np.concatenate([hellinger_distances, h], axis=0)
        classification_errors = np.concatenate([classification_errors, c], axis=0)
        jsd_losses = np.concatenate([jsd_losses, jsd], axis=0)

    return hellinger_distances, classification_errors, jsd_losses


def subsample_existing_classes(n_samples, class_counts, label_indices, distribution_p, classification_error_terms,
                               jsd_loss_terms, rng, num_classes, num_zero_classes, new_class_counts_min=None):
    new_class_counts_min = np.ones_like(class_counts) if new_class_counts_min is None else new_class_counts_min
    sampled_class_counts = np.array(
        [np.random.randint(n_min, n_k, n_samples) for n_min, n_k in zip(new_class_counts_min, class_counts)]
    ).T

    if num_zero_classes > 0:
        classes = np.arange(0, num_classes)
        zero_classes = np.array(
            [np.random.choice(classes, size=num_zero_classes, replace=False) for _ in range(n_samples)])
        for i in range(n_samples):
            sampled_class_counts[i, zero_classes[i, :]] = 0

    # compute hellinger distances
    sampled_distributions_q = sampled_class_counts / np.sum(sampled_class_counts, axis=1,
                                                            keepdims=True)
    distribution_p_tiled = np.tile(distribution_p, (n_samples, 1))
    squares = (np.sqrt(sampled_distributions_q) - np.sqrt(distribution_p_tiled)) ** 2
    hellinger_distances = np.sqrt(0.5 * np.sum(squares, axis=1))

    # sample indices for each class count vector
    classiciation_errors, jsd_losses = [], []
    for random_class_counts in tqdm(sampled_class_counts):
        indices = np.concatenate([
            rng.permutation(class_indices)[:n_k] for n_k, class_indices in zip(random_class_counts, label_indices)
        ], axis=0)

        # compute classification error loss
        classiciation_errors.append(np.mean(classification_error_terms[indices]))

        # compute jsd loss
        jsd_losses.append(np.mean(jsd_loss_terms[indices]))

    return hellinger_distances, classiciation_errors, jsd_losses


def compute_jsd_losses_per_sample(p, q):
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=1, keepdims=False)
    right_sum = np.sum(right, axis=1, keepdims=False)
    js = left_sum + right_sum
    js /= np.log(LOG_BASE)
    return js / 2.0


if __name__ == '__main__':
    rebuttal_phase = True
    if rebuttal_phase:
        cifar_data_root = 'results/data/cifar10/'
        predictions_roots = [
            cifar_data_root + 'densenet121',
            cifar_data_root + 'densenet169',
            cifar_data_root + 'googlenet',
            cifar_data_root + 'inception_v3',
            cifar_data_root + 'mobilenet_v2',
            cifar_data_root + 'resnet50',
            cifar_data_root + 'vgg11_bn',
            cifar_data_root + 'vgg19_bn',
        ]
        for pred_root in predictions_roots:
            do_sample(data_root=pred_root,
                      n_samples=100000,
                      num_zero_classes=list(np.arange(1, 9, step=1)),
                      num_add_classes=list(np.arange(1, 21, step=1)),
                      max_samples_per_new_class=1000,
                      frac_change_small_dist=[0.3],
                      max_remove_small_dist=[1000],
                      save_data=True)
    else:
        imagenet_specs = dict(data_root='results/data/imagenet/efficientnet_b7',
                              n_samples=20000,
                              num_zero_classes=list(np.arange(50, 500, step=50)),
                              num_add_classes=list(np.arange(50, 500, step=50)),
                              max_samples_per_new_class=50,
                              frac_change_small_dist=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                              max_remove_small_dist=[5, 10, 20, 30, 40, 50],
                              save_data=True)

        cifar_specs = dict(data_root='results/data/cifar10/cifar-resnet110',
                           n_samples=100000,
                           num_zero_classes=list(np.arange(1, 9, step=1)),
                           num_add_classes=list(np.arange(1, 21, step=1)),
                           max_samples_per_new_class=1000,
                           frac_change_small_dist=[0.3],
                           max_remove_small_dist=[1000],
                           save_data=False)

        # YELP (5-class)
        yelp_specs = dict(data_root='results/data/yelp/BERT_logits',
                          n_samples=100000,
                          num_zero_classes=list(np.arange(1, 4, step=1)),
                          num_add_classes=list(np.arange(1, 11, step=1)),
                          max_samples_per_new_class=500,
                          frac_change_small_dist=[0.3],
                          max_remove_small_dist=[500],
                          save_data=True)

        # SNLI
        snli_specs = dict(data_root='results/data/SNLI/DeBERTa_logits',
                          n_samples=100000,
                          num_zero_classes=list(np.arange(1, 2, step=1)),
                          num_add_classes=list(np.arange(1, 11, step=1)),
                          max_samples_per_new_class=3000,
                          frac_change_small_dist=[0.3],
                          max_remove_small_dist=[1000],
                          save_data=True)

        do_sample(**cifar_specs)
        do_sample(**yelp_specs)
        do_sample(**snli_specs)
        do_sample(**imagenet_specs)
