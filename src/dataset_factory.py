import numpy as np
import os
from typing import List

from torch.utils.data import Dataset
from torchvision import transforms

from constants import *
from lib.datasets.gaussian_sphere import GaussianSphere
from lib.datasets.gaussian_mixture import GaussianMixtureSmall, GaussianMixture
from lib.datasets.cifar10 import Cifar10, Cifar10Binary
from lib.datasets.imagenet import ImageNet, ImageNetBinary

__all__ = ["get_dataset", "get_num_classes"]


def get_dataset(dataset: str, split: str = None, sdev: float = -1., num_samples: int = 0, theta: np.ndarray = None,
                dim: int = -1, target_class_counts: List[int] = None, class_subset: List[int] = None,
                input_size: int = 224, num_classes=2, centers=None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == IMAGENET:
        return _imagenet(split, target_class_counts, input_size=input_size)
    elif dataset == IMAGENET_BINARY:
        return _imagenet_binary(split)
    elif dataset == IMAGENET_GRAYSCALE:
        return _imagenet_grayscale(split, input_size=input_size)
    elif dataset == CIFAR10:
        return _cifar10(split, target_class_counts)
    elif dataset == CIFAR10_BINARY:
        return _cifar10_binary(split, class_subset)
    elif dataset == GAUSSIAN_MIXUTRE:
        return _gaussian_mixture(sdev, num_samples, num_classes, dim, centers=centers)
    elif dataset == GAUSSIAN_SPHERE:
        return _gaussian_sphere(dim, sdev, num_samples, theta)

    raise ValueError(f'unknown dataset {dataset} !')


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset in [IMAGENET_GRAYSCALE, IMAGENET]:
        return 1000
    elif dataset == CIFAR10:
        return 10
    elif dataset == CIFAR10_BINARY or dataset == IMAGENET_BINARY or dataset == GAUSSIAN_SPHERE:
        return 2
    raise ValueError(f'unknown dataset {dataset}!')


def _cifar10(split: str, target_class_counts: List[int]) -> Dataset:
    if split == "train":
        return Cifar10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]), target_class_counts=target_class_counts)
    elif split == "test":
        return Cifar10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor(),
                       target_class_counts=target_class_counts)


def _cifar10_binary(split: str, class_subset: List[int]):
    if split == "train":
        return Cifar10Binary("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]), class_subset=class_subset)
    elif split == "test":
        return Cifar10Binary("./dataset_cache", train=False, download=True, transform=transforms.ToTensor(),
                             class_subset=class_subset)


def _imagenet(split: str, target_class_counts: List[int], input_size=224) -> Dataset:
    if IMAGENET_LOC_ENV not in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])
    else:
        raise ValueError
    return ImageNet(subdir, transform, target_class_counts=target_class_counts)


def _imagenet_binary(split: str) -> Dataset:
    if IMAGENET_LOC_ENV not in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    else:
        raise ValueError
    return ImageNetBinary(subdir, transform)


def _imagenet_grayscale(split: str, input_size=224) -> Dataset:
    if IMAGENET_LOC_ENV not in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomSizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Scale(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])
    else:
        raise ValueError
    return ImageNet(subdir, transform, target_class_counts=None)


def _gaussian_mixture(sdev, num_samples, num_classes, dim, centers=None):
    if num_classes > 2 or dim > 2:
        return GaussianMixture(sdev, num_samples, num_classes, dim, centers=centers)

    return GaussianMixtureSmall(sdev, num_samples)


def _gaussian_sphere(dim, sdev, num_samples, theta):
    return GaussianSphere(dim, sdev, num_samples, theta)
