import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax

from constants import *


def numpy_one_hot_encode(labels, epsilon=1e-6, num_classes=None):
    assert len(np.shape(labels)) == 1
    num_classes = labels.max() + 1 if num_classes is None else num_classes
    labels_one_hot = np.zeros((labels.size, num_classes)) + epsilon
    labels_one_hot[np.arange(labels.size), labels] = 1 - epsilon

    # normalize
    if epsilon > 0:
        labels_one_hot /= np.sum(labels_one_hot, axis=1, keepdims=True)

    return labels_one_hot


def euclidean_distance_to_hellinger(l2_perturbations, sdev):
    return np.sqrt(1 - np.exp(- l2_perturbations ** 2.0 / (8 * sdev ** 2)))


def compute_scores(logits, labels, func, num_classes=None, reduce=REDUCE_MEAN, return_count=False, predicted_probs=None):
    if func == CLASSIFICATION_ERROR:
        return _compute_classification_error(logits, labels, reduce=reduce, return_count=return_count)

    if func == JSD_LOSS:
        return _compute_jsd_loss(logits, labels, num_classes, reduce=reduce, return_count=return_count,
                                 predicted_probs=predicted_probs)

    if func == CLASSIFICATION_ACCURACY:
        return _compute_classification_accuracy_score(logits, labels, reduce=reduce, return_count=return_count)

    if func == AUC_SCORE:
        return _compute_auc_score(logits, labels, reduce=reduce, return_count=return_count, y_scores=predicted_probs)

    raise ValueError(f'unknown function {func}!')


def _compute_classification_error(logits, labels, reduce=REDUCE_MEAN, return_count=False):
    predicted_labels = np.argmax(logits, axis=1)
    loss_terms = np.array(predicted_labels != labels).astype(int)

    if return_count:
        return _reduce_terms(terms=loss_terms, reduce=reduce), len(loss_terms)

    return _reduce_terms(terms=loss_terms, reduce=reduce)


def _compute_jsd_loss(logits, labels, num_classes, reduce=REDUCE_MEAN, return_count=False, predicted_probs=None):
    one_hot_labels = numpy_one_hot_encode(labels, num_classes=num_classes)
    predicted_probs = softmax(logits, axis=1) if predicted_probs is None else predicted_probs
    loss_terms = jensenshannon(predicted_probs, one_hot_labels, axis=1, base=LOG_BASE) ** 2  # noqa

    if return_count:
        return _reduce_terms(terms=loss_terms, reduce=reduce), len(loss_terms)

    return _reduce_terms(terms=loss_terms, reduce=reduce)


def _compute_classification_accuracy_score(logits, labels, reduce=REDUCE_MEAN, return_count=False):
    predicted_labels = np.argmax(logits, axis=1)
    score_terms = np.array(predicted_labels == labels).astype(int)

    if return_count:
        return _reduce_terms(score_terms, reduce=reduce), len(score_terms)

    return _reduce_terms(score_terms, reduce=reduce)


def _compute_auc_score(logits, labels, reduce=REDUCE_MEAN, return_count=False, y_scores=None):
    """ first entry in logits corresponds to negative class, second entry to positive class """
    y_scores = softmax(logits, axis=1)[:, 1] if y_scores is None else y_scores[:, 1]

    if not all(np.unique(labels) == [0, 1]):
        _l = np.unique(labels)
        raise ValueError(f'Labels need to be binary and taking values in [0, 1] !, got {_l}')

    if np.shape(logits)[1] != 2:
        raise ValueError(f'logits have wrong number of classes! got shape {np.shape(logits)}')

    pos_indices = np.nonzero(labels == 1)[0]
    neg_indices = np.nonzero(labels == 0)[0]
    auc_scores = [int(si >= sj) for si in y_scores[pos_indices] for sj in y_scores[neg_indices]]

    if return_count:
        return _reduce_terms(terms=auc_scores, reduce=reduce), len(auc_scores)

    return _reduce_terms(terms=auc_scores, reduce=reduce)


def _reduce_terms(terms, reduce):
    if reduce == REDUCE_NONE:
        return terms

    if reduce == REDUCE_MEAN:
        return np.mean(terms)

    if reduce == REDUCE_VAR_UNBIASED:
        return np.var(terms, ddof=1)

    raise ValueError(f'unknow reduce method {reduce}!')
