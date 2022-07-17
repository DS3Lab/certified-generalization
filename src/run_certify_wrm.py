import json

import numpy as np
import os
import torch
import random

from constants import *
from lib.certify import CertifyWRM, CertifyGramian, CertifyLipschitz
from lib.certify.certify_utils import compute_scores, euclidean_distance_to_hellinger

ARCH = MLP_V2
USE_CUDA = torch.cuda.is_available()

# init seed
SEED = 742
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def compute_bounds(root_dir, arch):
    # load args
    with open(os.path.join(root_dir, 'args.json'), 'r') as f:
        args = json.load(f)

    # parse args
    num_hidden = args['n_hidden']
    width = args['width']
    num_classes = args.get('num_classes') or 2
    input_dim = args.get('input_dim') or 2

    # load data
    data = np.load(os.path.join(root_dir, 'data.npy'), allow_pickle=True)[()]
    covariates, labels = data['test_data']
    sdev = data['sdev']
    logits = data['test_logits']
    checkpoint = os.path.join(root_dir, 'checkpoint.pth.tar')

    # compute loss
    loss = compute_scores(logits=logits, labels=labels, func=JSD_LOSS, num_classes=num_classes)

    # distances
    euclidean_distances = np.linspace(0, 5.0, 50)
    hellinger_distances = euclidean_distance_to_hellinger(euclidean_distances, sdev)
    ws_distances = euclidean_distances ** 2

    # lipschitz certification
    certify_lipschitz = CertifyLipschitz(checkpoint, logits, labels, arch, finite_sampling=True, num_hidden=num_hidden,
                                         input_dim=input_dim, num_classes=num_classes, width=width)
    lipschitz_bounds = certify_lipschitz.certify(ws_distances=ws_distances)

    # wrm certification
    certify_wrm = CertifyWRM(checkpoint, arch, covariates, labels, finite_sampling=True, num_hidden=num_hidden,
                             input_dim=input_dim, num_classes=num_classes, width=width)
    wrm_bounds = certify_wrm.certify(ws_distances=ws_distances)

    certify_gramian = CertifyGramian(logits, labels, func=JSD_LOSS, finite_sampling=True)
    gramian_bounds = certify_gramian.certify(hellinger_distances)

    # save bounds
    np.save(os.path.join(root_dir, 'gamma.npy'), np.array(certify_wrm.gamma))
    np.save(os.path.join(root_dir, 'loss.npy'), np.array(loss))
    np.save(os.path.join(root_dir, 'euclidean_distances.npy'), euclidean_distances)
    np.save(os.path.join(root_dir, 'wrm_bounds.npy'), wrm_bounds)
    np.save(os.path.join(root_dir, 'gramian_bounds.npy'), gramian_bounds)
    np.save(os.path.join(root_dir, 'lipschitz_bounds.npy'), lipschitz_bounds)


if __name__ == '__main__':
    for rd in [
        'progression-h=2-w=2',
        'progression-h=2-w=4',
        'progression-h=2-w=8',
        'progression-h=2-w=16',
        'progression-h=5-w=2',
        'progression-h=10-w=2',
        'progression-h=20-w=2'
    ]:
        compute_bounds('results/data/gaussian-mixture/' + rd, ARCH)
