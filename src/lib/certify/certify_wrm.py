from collections.abc import Iterable
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from model_factory import get_architecture
from lib.utils.wrm_utils import adjust_lr_surrogate
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


class CertifyWRM:
    SUP_ITERATIONS = 200
    SUP_LR = 0.05
    L0 = 0.314568
    L1 = 0.5

    def __init__(self, checkpoint, arch: str, x: np.ndarray, y: np.ndarray, finite_sampling: bool = True, num_hidden: int = 1,
                 input_dim: int = 2, num_classes: int = 2, width: int = 2):
        self.model = _init_model(checkpoint, arch=arch, num_hidden=num_hidden, input_dim=input_dim,
                                 num_classes=num_classes, width=width)
        self.loss_fn = JSDLoss(num_classes=num_classes, reduce='mean')
        self.n_samples = len(y)
        self.m0 = FUNCS_ESS_SUP[JSD_LOSS]

        # init dataloader
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y).to(torch.int64)
        dataset = TensorDataset(tensor_x, tensor_y)
        self.dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

        # compute gamma
        self._gamma = self._compute_gamma()
        print(f'* gamma for {num_hidden} hidden layers and {width} width-multiplier: {self._gamma}')

        # finite sampling error
        if finite_sampling:
            self._fs_err_mean = self.m0 * np.sqrt(np.log(1.0 / CONFIDENCE_LEVEL) / (2.0 * self.n_samples))
        else:
            self._fs_err_mean = .0

    def certify(self, ws_distances) -> np.ndarray:
        wasserstein_distances = [ws_distances] if not isinstance(ws_distances, Iterable) else ws_distances
        surrogate_loss = self._compute_empirical_surrogate_loss()
        bounds = np.array([self._gamma * rho + surrogate_loss + self._fs_err_mean for rho in wasserstein_distances])
        return bounds

    def _compute_empirical_surrogate_loss(self):
        # compute surrogate loss
        total_surrogate_loss = .0
        for x_batch, y_batch in self.dataloader:
            x_batch = torch.autograd.Variable(x_batch)
            y_batch = torch.autograd.Variable(y_batch)
            batch_size = len(y_batch)

            # compute surrogate loss (= inner sup)
            surrogate_loss, _, _ = self._surrogate_loss_batch(x_batch, y_batch)  # loss is sum of individual losses
            total_surrogate_loss -= surrogate_loss * batch_size

        # divide by number of samples
        mean_surrogate_loss = total_surrogate_loss / self.n_samples
        return mean_surrogate_loss

    def _surrogate_loss_batch(self, x_batch, y_batch):
        z_batch = x_batch.data.clone()
        z_batch = z_batch.cuda() if USE_CUDA else z_batch
        z_batch = torch.autograd.Variable(z_batch, requires_grad=True)

        # run inner optimization
        surrogate_optimizer = optim.Adam([z_batch], lr=self.SUP_LR)
        surrogate_loss = .0  # phi(theta,z0)
        rho = .0  # E[c(Z,Z0)]
        for t in range(self.SUP_ITERATIONS):
            surrogate_optimizer.zero_grad()
            distance = z_batch - x_batch
            rho = torch.mean((torch.norm(distance.view(len(x_batch), -1), 2, 1) ** 2))
            loss_zt = self.loss_fn(self.model(z_batch.float()), y_batch)
            surrogate_loss = - (loss_zt - self._gamma * rho)
            surrogate_loss.backward()
            surrogate_optimizer.step()
            adjust_lr_surrogate(surrogate_optimizer, self.SUP_LR, t + 1)

        return surrogate_loss.data, rho.data, z_batch

    def _compute_gamma(self):
        # compute operator norm
        weight_mats = [p.detach() for p in self.model.parameters()]
        operator_norms = []

        for w in weight_mats:
            operator_norms.append(torch.linalg.matrix_norm(w, 2))

        alpha_values = np.cumprod(operator_norms)
        beta = alpha_values[-1] * np.sum(alpha_values)
        gamma = self.L0 * beta + self.L1 * alpha_values[-1] ** 2
        return gamma

    @property
    def gamma(self):
        return self._gamma
