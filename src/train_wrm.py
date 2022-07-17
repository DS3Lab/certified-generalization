import argparse
import copy
import json
import numpy as np
import random
from tqdm import trange
import os

import torch
from torch.utils import data as data_utils
import torch.optim as optim
from dataset_factory import get_dataset

from lib.utils.wrm_utils import adjust_lr_surrogate
from lib.utils.train_utils import directory_setup
from lib.loss_functions import JSDLoss
from model_factory import get_architecture
from constants import MLP_V2, GAUSSIAN_MIXUTRE

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--n_hidden', default=2, type=int, help='number hidden layers in the MLP model')
parser.add_argument('--num_classes', default=2, type=int, help='number of centers in the Gaussian Mixture')
parser.add_argument('--input_dim', default=2, type=int, help='dimenstionality of the data')
parser.add_argument('--width', type=int, default=2)
parser.add_argument('--gamma', type=float, default=2, help='lagrangian multiplier in innner maximization')

parser.add_argument('--outdir', default='./results/data', type=str, help='folder to save model and training log)')
parser.add_argument('--ntrain', default=5000, type=int, help='number of samples in the training set')
parser.add_argument('--ntest', default=5000, type=int, help='number of samples in the test set')
parser.add_argument('--id', default='progression', type=str, help='id of train run')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--steps-surrogate', default=20, type=int, help='steps in inner optimization')
parser.add_argument('--batch', default=500, type=int, metavar='N', help='batchsize (default: 500)')
parser.add_argument('--lr0', default=0.01, type=float, help='learning rate for outer minimization')
parser.add_argument('--lr1', default=0.08, type=float, help='learning rate for inner maximimization')

args = parser.parse_args()

# constant hyperparams
SDEV = 1.0
ACTIV_FN = 'elu'

USE_CUDA = torch.cuda.is_available()

# init seed
SEED = 743
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def train():
    dataset_name = 'gaussian-mixture'

    if args.num_classes > 2 or args.input_dim > 2:
        dataset_name = dataset_name + f'-c={args.num_classes}-d={args.input_dim}'

    # setup dir structure
    train_id = args.id + f'-h={args.n_hidden}-w={args.width}'
    results_dir = directory_setup(train_id=train_id, dataset=dataset_name, results_dir=args.outdir)
    print(f'saving results to {results_dir}')

    # dump args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # generate data
    train_dataset = get_dataset(GAUSSIAN_MIXUTRE, 'train', SDEV, args.ntrain, dim=args.input_dim,
                                num_classes=args.num_classes, centers=None)
    test_dataset = get_dataset(GAUSSIAN_MIXUTRE, 'test', SDEV, args.ntest, dim=args.input_dim,
                               num_classes=args.num_classes, centers=train_dataset.centers)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # setup model and loss
    model = get_architecture(arch=MLP_V2, dataset=GAUSSIAN_MIXUTRE, activ_fn=ACTIV_FN, num_hidden_mlp=args.n_hidden,
                             num_classes=args.num_classes, input_dim=args.input_dim, width_multiplier=args.width)
    model = model.cuda() if USE_CUDA else model

    criterion = JSDLoss(num_classes=args.num_classes)

    # setup optimizer and lr scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr0)

    # training history
    train_hist = dict(total_loss=[], surrogate_loss=[], acc_train=[], acc_test=[])

    # train loop
    epoch_bar = trange(args.epochs, leave=True)
    for epoch in epoch_bar:
        total_losses, surrogate_losses, rho_values = train_epoch(model=model,
                                                                 dataloader=train_dataloader,
                                                                 optimizer=optimizer,
                                                                 criterion=criterion,
                                                                 lr_surrogate=args.lr1,
                                                                 steps_surrogate=args.steps_surrogate,
                                                                 gamma_surrogate=args.gamma)

        total_loss = torch.mean(torch.FloatTensor(total_losses))  # E(l(theta,Z))
        surrogate_loss = torch.mean(torch.FloatTensor(surrogate_losses))  # E[phi_gamma(theta,Z)]
        distance_loss = torch.mean(torch.FloatTensor(rho_values))  # E[c(Z,Z0)]

        # evaluate train and test accuracy
        acc_train = evaluate(model, train_dataloader)
        acc_test = evaluate(model, test_dataloader)

        train_hist['total_loss'].append(total_loss)
        train_hist['surrogate_loss'].append(-surrogate_loss)
        train_hist['acc_train'].append(acc_train)
        train_hist['acc_test'].append(acc_test)

        # update progress bar
        bar_descr = f"[epoch {epoch}] loss: {total_loss:.3f} train acc: {acc_train:.1f}% test acc: {acc_test:.1f}% "
        bar_descr += f"surrogate loss: {surrogate_loss}, dist loss: {distance_loss}"
        epoch_bar.set_description(bar_descr)
        epoch_bar.refresh()

        # save model
        state_dict = {'model': model.cpu().state_dict() if USE_CUDA else model.state_dict(), 'train_hist': train_hist}
        torch.save(copy.deepcopy(state_dict), os.path.join(results_dir, 'checkpoint.pth.tar'))
        model = model.cuda() if USE_CUDA else model

    # compute logits on testing data
    test_logits, test_labels = compute_logits(model, test_dataloader)

    # compute logits on training data
    train_logits, train_labels = compute_logits(model, train_dataloader)

    # save data
    data = {
        'test_logits': test_logits,
        'train_logits': train_logits,
        'test_labels': test_labels,
        'train_labels': train_labels,
        'sdev': SDEV,
        'train_data': train_dataset.data,
        'test_data': test_dataset.data
    }
    np.save(os.path.join(results_dir, 'data.npy'), data)


def train_epoch(model, dataloader, optimizer, criterion, lr_surrogate, steps_surrogate, gamma_surrogate):
    model.train()
    total_losses = []
    surrogate_losses = []
    rho_values = []

    for x_batch, y_batch in dataloader:
        if USE_CUDA:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        # compute surrogate loss (= inner sup)
        surrogate_loss, rho, z_batch = compute_surrogate_loss(model=model, x_batch=x_batch, y_batch=y_batch,  # noqa
                                                              lr0=lr_surrogate, num_steps=steps_surrogate,
                                                              loss_function=criterion, gamma=gamma_surrogate)

        # run outer optimization step
        optimizer.zero_grad()
        total_loss = criterion(model(z_batch.float()), y_batch)
        total_loss.backward()
        optimizer.step()

        surrogate_losses.append(surrogate_loss.cpu())
        total_losses.append(total_loss.data.cpu())
        rho_values.append(rho.cpu())

    return total_losses, surrogate_losses, rho_values


def compute_surrogate_loss(model, x_batch, y_batch, lr0, num_steps, loss_function, gamma):
    z_batch = x_batch.data.clone()
    z_batch = z_batch.cuda() if USE_CUDA else z_batch
    z_batch = torch.autograd.Variable(z_batch, requires_grad=True)

    # run inner optimization
    surrogate_optimizer = optim.Adam([z_batch], lr=lr0)
    surrogate_loss = .0  # phi(theta,z0)
    rho = .0  # E[c(Z,Z0)]
    for t in range(num_steps):
        surrogate_optimizer.zero_grad()
        distance = z_batch - x_batch
        rho = torch.mean((torch.norm(distance.view(len(x_batch), -1), 2, 1) ** 2))
        loss_zt = loss_function(model(z_batch.float()), y_batch)
        surrogate_loss = - (loss_zt - gamma * rho)
        surrogate_loss.backward()
        surrogate_optimizer.step()
        adjust_lr_surrogate(surrogate_optimizer, lr0, t + 1)

    return surrogate_loss.data, rho.data, z_batch


def evaluate(model, dataloader):
    model.eval()
    counter, acc = .0, .0
    for x_batch, y_batch in dataloader:
        if USE_CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        out = model(x_batch.float())
        _, predicted = torch.max(out, 1)
        counter += y_batch.size(0)
        acc += float(torch.eq(predicted, y_batch).sum().cpu().data.numpy())

    acc = acc / float(counter) * 100.0
    return acc


def compute_logits(model, dataloader):
    model.eval()
    logits = np.empty(shape=(0, dataloader.dataset.num_classes))
    labels = np.empty(shape=0)
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if USE_CUDA:
            x_batch = x_batch.cuda()

        with torch.no_grad():
            batch_logits = model(x_batch.float()).cpu().numpy()
        logits = np.concatenate([logits, batch_logits])
        labels = np.concatenate([labels, y_batch])

    return logits, labels


if __name__ == '__main__':
    train()
