from collections.abc import Iterable
import torch
import numpy as np

from model_factory import get_architecture
from lib.certify.certify_utils import compute_scores
from lib.loss_functions import JSDLoss
from constants import *

USE_CUDA = torch.cuda.is_available()


def _init_model(ckpt, arch, num_hidden, input_dim=2, num_classes=2, width=2):
    model = get_architecture(arch=arch, dataset=GAUSSIAN_MIXUTRE, activ_fn='elu', num_hidden_mlp=num_hidden,
                             input_dim=input_dim, num_classes=num_classes, width_multiplier=width)
    checkpoint = torch.load(ckpt)
    model_params = checkpoint['model']
    model.load_state_dict(model_params)
    return model


class CertifyLipschitz:
    SOFTMAX_JSD_LIPSCH_CONST = 0.314568

    def __init__(self, checkpoint, logits, labels, arch: str, finite_sampling: bool = True, num_hidden: int = 1,
                 input_dim: int = 2, num_classes: int = 2, width: int = 2):
        self.logits = logits
        self.labels = labels
        self.num_classes = np.max(labels) + 1

        self.n_samples = len(labels)
        self.loss_fn = JSDLoss(num_classes=num_classes, reduce='mean')
        self.func_ess_sup = FUNCS_ESS_SUP[JSD_LOSS]

        self.model = _init_model(checkpoint, arch, num_hidden, input_dim, num_classes, width)

        # compute the Lipschitz constant of our NN
        self._lipschitz_constant = self._compute_lipschitz_constant()
        print(f'* Lipschitz constant for n_h={num_hidden} and width-multiplier={width}: {self._lipschitz_constant}')

        # finite sampling error
        if finite_sampling:
            self._fs_err_mean = self.func_ess_sup * np.sqrt(np.log(1.0 / CONFIDENCE_LEVEL) / (2.0 * self.n_samples))
        else:
            self._fs_err_mean = .0

    def certify(self, ws_distances) -> np.ndarray:
        wasserstein_distances = [ws_distances] if not isinstance(ws_distances, Iterable) else ws_distances
        mean = compute_scores(self.logits, self.labels, JSD_LOSS, self.num_classes, REDUCE_MEAN, return_count=False)
        bounds = np.array([self._lipschitz_constant * rho + mean + self._fs_err_mean for rho in wasserstein_distances])
        return bounds

    def _compute_lipschitz_constant(self):
        """
        assumption: activation functions have Lipschitz constant 1 (holds for elu, relu, ...)
        """
        weights = [p.detach() for p in self.model.parameters()]
        norms = [torch.linalg.matrix_norm(w, 2) for w in weights]  # 2 is largest singular value (= spectral norm)
        return np.prod(norms) * self.SOFTMAX_JSD_LIPSCH_CONST

    @property
    def lipschitz_constant(self):
        return self._lipschitz_constant
