import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    """Dice loss of binary class
    Args:
    Returns:
        Loss tensor
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        target = y.type(torch.LongTensor).cuda()
        loss_c = nn.CrossEntropyLoss()(x, target)
        _, classification_pred = torch.max(x, 1)
        acc = (classification_pred == target).sum().type(torch.FloatTensor)

        return loss_c,


class SegmentationLoss(nn.Module):
    """Dice loss of binary class
    Args:
    Returns:
        Loss tensor
    """
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss()

    def __len__(self):
        return 1

    def forward(self, x, y):
        x = x[0]
        x = x.permute(0, 2, 3, 1).cuda()
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        y = y.view(-1)

        loss_cross_entropy, = self.cross_entropy_loss(x, y)

        loss_all = [loss_cross_entropy]
        loss_val = 1 * loss_all[0]

        return loss_val, loss_all

