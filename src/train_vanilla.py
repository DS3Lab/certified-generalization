# this code is based on publicly available code at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
import os
import argparse
import json
import datetime
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from constants import *
from dataset_factory import get_dataset, get_num_classes
from model_factory import get_architecture
from lib.utils.train_utils import AverageMeter, accuracy, init_logfile, log, directory_setup
from lib.loss_functions import JSDLoss

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS, default=None)
parser.add_argument('arch', type=str, choices=ARCHITECTURES, default=None)
parser.add_argument('--init-pretrained', default='0', type=int, choices=[0, 1])
parser.add_argument('--unbalanced', default='0', type=int, choices=[0, 1])
parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'jsd'])
parser.add_argument('--outdir', default='./temp', type=str, help='folder to save model and training log)')
parser.add_argument('--id', type=str, default=None, help='folder to save model and training log)')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N', help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30, help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
NUM_CLASSES = get_num_classes(args.dataset)

SEED = 742

# init seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def main():
    # setup dir structure
    results_dir = directory_setup(train_id=args.id, dataset=args.dataset, results_dir=args.outdir)

    # dump args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # make dataset unnablanced
    if args.dataset == CIFAR10:
        target_class_counts = CIFAR10_CLASS_COUNTS_UNBALANCED if args.unbalanced else CIFAR10_CLASS_COUNTS
    elif args.dataset == IMAGENET:
        target_class_counts = IMAGENET_CLASS_COUNTS_UNBALANCED if args.unbalanced else IMAGENET_CLASS_COUNTS
    else:
        target_class_counts = None

    # dump class counts
    if target_class_counts is not None:
        np.save(os.path.join(results_dir, 'class-counts.npy'), np.array(target_class_counts, dtype=int))

    # fetch datasets and dataloaders
    train_dataset = get_dataset(args.dataset, 'train', target_class_counts=target_class_counts)
    test_dataset = get_dataset(args.dataset, 'test', target_class_counts=target_class_counts)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers,
                             pin_memory=args.dataset == "imagenet")

    # build model
    model = get_architecture(args.arch, args.dataset, load_weights=args.init_pretrained, use_cuda=USE_CUDA)

    # logging
    logfilename = os.path.join(results_dir, 'log.txt')
    log_columns = "epoch\ttime\tlr\ttrain-total-loss\ttrain-mean-loss\ttrain-var-loss\ttrain-acc\ttest-total-loss"
    log_columns += "\ttest-mean-loss\ttest-var-loss\ttest-acc"
    init_logfile(logfilename, log_columns)

    num_classes = get_num_classes(args.dataset)
    if args.loss == 'ce':
        criterion = CrossEntropyLoss().cuda()
    elif args.loss == 'jsd':
        criterion = JSDLoss(num_classes=num_classes).cuda()
    else:
        raise ValueError(f'unknown loss function {args.loss}')

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    print('start training')

    for epoch in range(args.epochs):
        before = time.time()
        train_total_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        test_total_loss, test_acc = test(test_loader, model, criterion)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_last_lr()[0], train_total_loss, train_acc, test_total_loss, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(results_dir, 'checkpoint.pth.tar'))

        scheduler.step()

    # compute logits on testing data
    test_logits, test_labels = compute_logits(model, test_loader, num_classes)

    # save data
    if args.dataset == IMAGENET:
        data = {'test_logits': test_logits, 'test_labels': test_labels}
    else:
        # compute logits on training data (only if not imagenet)
        train_logits, train_labels = compute_logits(model, train_loader, num_classes)
        data = {'test_logits': test_logits, 'train_logits': train_logits,
                'test_labels': test_labels, 'train_labels': train_labels}

    np.save(os.path.join(results_dir, 'data.npy'), data)


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        outputs = model(inputs)

        # compute loss
        total_loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5) if NUM_CLASSES >= 5 else (1, 2))
        total_losses.update(total_loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # set grads to zero
        for param in model.parameters():
            param.grad = None

        # compute grads and make train step
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(loader), batch_time=batch_time,
                                                                 data_time=data_time, total_loss=total_losses,
                                                                 top1=top1, top5=top5))
    return total_losses.avg, top1.avg


def test(loader: DataLoader, model: torch.nn.Module, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if USE_CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # compute output
            outputs = model(inputs)

            # compute loss
            total_loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5) if NUM_CLASSES >= 5 else (1, 2))
            total_losses.update(total_loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(loader), batch_time=batch_time,
                                                                     data_time=data_time, total_loss=total_losses,
                                                                     top1=top1, top5=top5))

        return total_losses.avg, top1.avg


def compute_logits(model, dataloader, num_classes):
    model.eval()
    logits = np.empty(shape=(0, num_classes))
    labels = np.empty(shape=0)
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if USE_CUDA:
            x_batch = x_batch.cuda()

        with torch.no_grad():
            batch_logits = model(x_batch.float()).cpu().numpy()
        logits = np.concatenate([logits, batch_logits])
        labels = np.concatenate([labels, y_batch])

    return logits, labels


if __name__ == "__main__":
    main()
