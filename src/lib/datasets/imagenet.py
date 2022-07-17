import numpy as np
from typing import List
from torchvision import datasets


class ImageNet(datasets.ImageFolder):
    def __init__(self, subdir, transform, target_class_counts: List[int] = None):
        super(ImageNet, self).__init__(subdir, transform)
        if target_class_counts is None:
            return

        _, class_counts = np.unique(self.targets, return_counts=True)
        target_class_counts = np.array(target_class_counts)

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
        self.targets = [self.targets[i] for i in label_indices_subsampled]
        self.samples = [self.samples[i] for i in label_indices_subsampled]

    @property
    def num_classes(self) -> int:
        return 1000


class ImageNetBinary(datasets.ImageFolder):
    CLASS_SUBSET = [20, 827]

    def __init__(self, subdir, transform):
        super(ImageNetBinary, self).__init__(subdir, transform)
        indices_subset = np.array(
            [i for i in range(len(self.targets)) if self.targets[i] in self.CLASS_SUBSET]).astype(int)

        samples = [self.samples[i] for i in indices_subset]
        targets = [self.targets[i] for i in indices_subset]

        # convert to 0 1 index
        convert_class = {cl: i for i, cl in enumerate(self.CLASS_SUBSET)}
        self.samples = list(map(lambda x: (x[0], convert_class[x[1]]), samples))
        self.targets = list(map(lambda x: convert_class[x], targets))

    @property
    def num_classes(self):
        return 2
