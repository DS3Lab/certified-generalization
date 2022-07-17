import torch
import torch.nn.functional as F  # noqa


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log2() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, dim=1)
    return output


class JSDLoss(torch.nn.Module):
    def __init__(self, num_classes, reduce='mean'):
        super(JSDLoss, self).__init__()
        self.num_classes = num_classes
        self.reduce = reduce

    def set_reduce(self, reduce):
        self.reduce = reduce

    def forward(self, predictions, labels):
        preds = F.softmax(predictions, dim=1)
        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels, preds]
        mean_distrib = sum(distribs) / len(distribs)
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log2()

        kldivs1 = custom_kl_div(mean_distrib_log, labels)
        kldivs2 = custom_kl_div(mean_distrib_log, preds)

        if self.reduce == 'mean':
            return 0.5 * (kldivs1.mean() + kldivs2.mean())
        if self.reduce == 'none':
            return 0.5 * (kldivs1 + kldivs2)
        if self.reduce == 'sum':
            return 0.5 * (kldivs1.sum() + kldivs2.sum())

        assert False
