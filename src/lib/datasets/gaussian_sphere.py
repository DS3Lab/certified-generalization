import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["GaussianSphere"]

_PI0 = 0.9


class GaussianSphere(Dataset):
    def __init__(self, dim, sdev, n, theta=None):
        """
        Class to generate data on a sphere which is distributed according to the model

            X ~ N(0, sdev^2 1d)
            P(Y = +1 | X=x) = pi * 1{sign(theta^T * x)=1} + (1 - pi) * 1{sign(theta^T * x) = -1}

        where theta is sampled uniformly at random from the sphere
        :param dim: input dimension
        :param sdev: standard deviation of the covariates
        :param n: number of samples
        """
        # generate covariates
        cov_mat = sdev ** 2 * np.identity(dim)
        self._x = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov_mat, size=n)

        # sample vector theta
        if theta is None:
            self._theta = sample_spherical(1, ndim=dim)
        else:
            self._theta = theta

        # generate labels
        signs = np.sign(np.matmul(self._x, self._theta))
        bernoulli_probs = [_PI0 if s == -1 else 1 - _PI0 for s in signs]
        self._y = np.array([np.random.binomial(1, p) for p in bernoulli_probs])

    @property
    def theta(self):
        return self._theta

    @property
    def data(self):
        return self._x, self._y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx], self._y[idx]


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
