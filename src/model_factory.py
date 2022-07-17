from typing import *
import torch
from torchvision.models.efficientnet import efficientnet_b7
from torchvision.models.resnet import resnet50, resnet152
import torch.backends.cudnn as cudnn

from lib.architectures.mlp import NeuralNetV2
from lib.architectures.cifar_resnet import resnet as resnet_cifar
from constants import *

__all__ = ["get_architecture"]

_IMAGENET_MODELS = {RESNET50: resnet50, RESNET152: resnet152, EFFICENTNETB7: efficientnet_b7}
_IMAGENET_BINARY_MODELS = {RESNET50: lambda: resnet50(num_classes=2),
                           RESNET152: lambda: resnet152(num_classes=2),
                           EFFICENTNETB7: lambda: efficientnet_b7(num_classes=2)}
_CIFAR_MODELS = {CIFAR_RESNET20: lambda x: resnet_cifar(depth=20, num_classes=x),
                 CIFAR_RESNET110: lambda x: resnet_cifar(depth=110, num_classes=x)}

# dataset constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def get_architecture(arch: str, dataset: str, load_weights: bool = False, use_cuda: bool = False,
                     activ_fn: str = 'elu', num_hidden_mlp=1, input_dim=2, num_classes=2,
                     width_multiplier=4) -> torch.nn.Module:
    """ Return a neural network (with random imagenet) """
    if dataset == IMAGENET or dataset == IMAGENET_GRAYSCALE:
        model = _IMAGENET_MODELS[arch](pretrained=load_weights)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    elif dataset == IMAGENET_BINARY:
        model = _IMAGENET_BINARY_MODELS[arch]()
        model = torch.nn.DataParallel(model)
    elif dataset == CIFAR10:
        model = _CIFAR_MODELS[arch](10)
    elif dataset == CIFAR10_BINARY:
        model = _CIFAR_MODELS[arch](2)
    elif dataset == GAUSSIAN_MIXUTRE:
        model = NeuralNetV2(activation=activ_fn, num_hidden=num_hidden_mlp, input_dim=input_dim,
                            num_classes=num_classes, width_multiplier=width_multiplier)
    else:
        raise ValueError(f'unknown arch {arch}')

    if use_cuda:
        model = model.cuda()

    normalize_layer = get_normalize_layer(dataset, use_cuda)
    return torch.nn.Sequential(normalize_layer, model)


def get_normalize_layer(dataset: str, use_cuda: bool) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == IMAGENET or dataset == IMAGENET_BINARY or dataset == IMAGENET_GRAYSCALE:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, use_cuda)
    elif dataset == CIFAR10 or dataset == CIFAR10_BINARY:
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, use_cuda)
    elif dataset in [GAUSSIAN_SPHERE, GAUSSIAN_MIXUTRE]:
        return torch.nn.Identity()
    else:
        raise ValueError(f"unknown dataset {dataset}!")


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float], use_cuda: bool):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)
        if use_cuda:
            self.means = self.means.cuda()
            self.sds = self.sds.cuda()

    def forward(self, inputs: torch.tensor):
        (batch_size, num_channels, height, width) = inputs.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (inputs - means) / sds
