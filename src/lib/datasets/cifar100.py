import numpy as np
from torchvision import datasets


class Cifar100(datasets.CIFAR100):
    def __init__(self, root, train, download, transform, subsample_indices=None):
        super(Cifar100, self).__init__(root=root, train=train, download=download, transform=transform)
        if subsample_indices is not None:
            self.data = self.data[subsample_indices, :]
            self.targets = list(np.array(self.targets)[subsample_indices])
