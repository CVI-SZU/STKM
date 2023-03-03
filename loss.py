import torch
from torch import nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    # Implement label smoothing.

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask1 = torch.nonzero(target.data == self.padding_idx)
        mask2 = torch.nonzero(target.data == 1)

        if mask1.dim() > 0:
            true_dist.index_fill_(0, mask1.squeeze(), 0.0)
        if mask2.dim() > 0:
            true_dist.index_fill_(0, mask2.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

