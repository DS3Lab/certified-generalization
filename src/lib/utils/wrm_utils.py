import numpy as np

import torch
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()


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


def adjust_lr_surrogate(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, lr0, epoch, total_epochs):
    lr = lr0 * (0.1 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
