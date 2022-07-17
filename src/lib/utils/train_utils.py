import uuid
import os
import torch


def directory_setup(train_id, dataset, results_dir=None):
    if train_id is None:
        train_id = 'id-' + str(uuid.uuid4())
        print(f'==> generated random uuid {train_id}')

    results_dir = os.path.join(results_dir, dataset, str(train_id))

    try:
        os.makedirs(results_dir)
    except OSError:
        raise OSError(f'results dir already exists! {results_dir}')

    return results_dir


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text + "\n")
    f.close()


def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text + "\n")
    f.close()
