import numpy as np

from lib.certify.certify_utils import compute_scores
from constants import *


class CertifyGramian:
    def __init__(self, logits, labels, func=JSD_LOSS, finite_sampling=True):
        self.logits = logits
        self.labels = labels
        self.func = func
        self.finite_sampling = finite_sampling

        self.num_classes = np.max(labels) + 1
        self.func_ess_sup = FUNCS_ESS_SUP[func]
        self.func_ess_inf = FUNCS_ESS_INF[func]

    def certify(self, distances, upper_bound=True):
        # compute sample mean and variance
        mean, n = compute_scores(self.logits, self.labels, self.func, self.num_classes, REDUCE_MEAN, return_count=True)
        variance = compute_scores(self.logits, self.labels, self.func, self.num_classes, REDUCE_VAR_UNBIASED)

        # compute certified bounds for each rho
        if upper_bound:
            return np.array([self._certify_upper_bound_at_rho(mean, variance, rho, n) for rho in distances])

        return np.array([self._certify_lower_bound_at_rho(mean, variance, rho, n) for rho in distances])

    def _certify_upper_bound_at_rho(self, m, v, rho, n):
        # account for finite sampling error with union bound (Hoeffding + Maurer)
        if self.finite_sampling:
            width = self.func_ess_sup - self.func_ess_inf
            m = min(1, m + width * np.sqrt(np.log(2 / CONFIDENCE_LEVEL) / (2 * n)))
            v = (np.sqrt(v) + width * np.sqrt(2 * np.log(2 / CONFIDENCE_LEVEL) / (n - 1))) ** 2

        rho = rho ** 2
        if rho >= 1 - 1.0 / np.sqrt(1 + (self.func_ess_sup - m) ** 2 / v):
            return self.func_ess_sup
        c = np.sqrt(rho * (2 - rho) * (1 - rho) ** 2)
        return m + 2 * c * np.sqrt(v) + rho * (2 - rho) * (self.func_ess_sup - m - v / (self.func_ess_sup - m))

    def _certify_lower_bound_at_rho(self, m, v, rho, n):
        # account for finite sampling error
        if self.finite_sampling:
            width = self.func_ess_sup - self.func_ess_inf
            m = max(0, m - width * np.sqrt(np.log(2 / CONFIDENCE_LEVEL) / (2 * n)))
            v = (np.sqrt(v) + width * np.sqrt(2 * np.log(2 / CONFIDENCE_LEVEL) / (n - 1))) ** 2

        rho = rho ** 2
        x = m ** 2 / v
        if rho > 1 - np.sqrt(1.0 / (1.0 + x)):
            return .0
        c = np.sqrt(rho * (2 - rho) * (1 - rho) ** 2)
        return m - 2 * c * np.sqrt(v) - rho * (2 - rho) * (m - v / m)
