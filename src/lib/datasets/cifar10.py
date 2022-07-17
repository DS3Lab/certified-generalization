import numpy as np
from typing import List
from torchvision import datasets


class Cifar10(datasets.CIFAR10):
    """
    this class inherits from the classical CIFAR10 dataset class and enables to subsample according to special
    class counts
    """
    def __init__(self, root, train, download, transform, target_class_counts: List[int] = None):
        super(Cifar10, self).__init__(root=root, train=train, download=download, transform=transform)
        if target_class_counts is None:
            return

        _, class_counts = np.unique(self.targets, return_counts=True)
        target_class_counts = np.array(target_class_counts)

        self._subsample_indices(class_counts, target_class_counts)

    def _subsample_indices(self, class_counts, target_class_counts):
        if not np.array(class_counts >= target_class_counts).all():
            # adjust class counts
            pivot_ratio = np.max(target_class_counts.astype(float) / class_counts.astype(float)) ** (-1)
            target_class_counts = np.floor(float(pivot_ratio) * target_class_counts).astype(int)

        # randomly subsample indices
        targets = np.array(self.targets, dtype=int)
        label_indices = np.array([np.squeeze(np.nonzero(targets == k)) for k in range(self.num_classes)])
        label_indices_subsampled = [np.random.choice(idx, n) for idx, n in zip(label_indices, target_class_counts)]
        label_indices_subsampled = np.concatenate(label_indices_subsampled)

        # slice data
        self.targets = list(targets[label_indices_subsampled])
        self.data = self.data[label_indices_subsampled, :]

    @property
    def num_classes(self) -> int:
        return 10


class Cifar10Binary(datasets.CIFAR10):
    def __init__(self, root, train, download, transform, class_subset: List[int] = None):
        super(Cifar10Binary, self).__init__(root=root, train=train, download=download, transform=transform)
        if class_subset is None:
            class_subset = [2, 7]  # default binarized if None
        indices_subset = np.array([i for i in range(len(self.targets)) if self.targets[i] in class_subset]).astype(int)
        targets = np.array(self.targets)[indices_subset]
        self.targets = list(map(lambda x: {cl: i for i, cl in enumerate(class_subset)}[x], targets))  # convert to 0, 1
        self.data = self.data[indices_subset, :]
