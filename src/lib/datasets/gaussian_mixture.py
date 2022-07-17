import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["GaussianMixtureSmall", "GaussianMixture"]


class GaussianMixtureSmall(Dataset):
    N_CLASSES = 2

    def __init__(self, sdev, n):
        """
        Class to generate data sampled from a Gaussian mixture

        where theta is sampled uniformly at random from the sphere
        # :param n_classes: number of classification categories
        # :param dim: input dimension
        :param sdev: standard deviation of the covariates
        :param n: number of samples
        """
        # generate centers
        center_pos = np.array([2, 0])
        center_neg = np.array([-2, 0])
        self._centers = [center_pos, center_neg]

        # covariance matrix
        cov_mat = sdev ** 2 * np.identity(2)

        # generate Gaussian blob for each class
        n_samples_per_class = n // 2
        samples_pos = np.random.multivariate_normal(mean=center_pos, cov=cov_mat, size=n_samples_per_class)
        samples_neg = np.random.multivariate_normal(mean=center_neg, cov=cov_mat, size=n_samples_per_class)
        self._x = np.concatenate([samples_pos, samples_neg], axis=0, dtype=float)
        self._y = np.concatenate([np.zeros(n_samples_per_class, dtype=int),
                                  np.ones(n_samples_per_class, dtype=int)])

    @property
    def data(self):
        return self._x, self._y

    @property
    def num_classes(self):
        return self.N_CLASSES

    @property
    def centers(self):
        return self._centers

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx], self._y[idx]


class GaussianMixture(Dataset):

    def __init__(self, sdev, n, num_centers, n_dim, dist_multiplier=2, centers=None):
        """
        Class to generate data sampled from a Gaussian mixture

        where theta is sampled uniformly at random from the sphere
        # :param n_classes: number of classification categories
        # :param dim: input dimension
        :param sdev: standard deviation of the covariates
        :param n: number of samples
        """
        self._num_classes = num_centers
        # generate centers
        assert num_centers <= 2 * n_dim

        if centers is None:
            centers = [dist_multiplier * sdev * np.eye(1, n_dim, i).reshape(-1) for i in range(n_dim)]
            centers = centers + [-dist_multiplier * sdev * np.eye(1, n_dim, i).reshape(-1) for i in range(n_dim)]
            centers = list(np.array(centers)[np.random.choice(range(len(centers)), num_centers, replace=False)])

        self._centers = centers

        # covariance matrix
        cov_mat = sdev ** 2 * np.identity(n_dim)

        # generate Gaussian cluster for each class
        n_samples_per_class = n // num_centers
        self._x = np.concatenate([
            np.random.multivariate_normal(mean=c, cov=cov_mat, size=n_samples_per_class) for c in centers
        ], axis=0, dtype=float)
        self._y = np.concatenate([np.ones(n_samples_per_class, dtype=int) * k for k in range(num_centers)])

    @property
    def data(self):
        return self._x, self._y

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def centers(self):
        return self._centers

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx], self._y[idx]
