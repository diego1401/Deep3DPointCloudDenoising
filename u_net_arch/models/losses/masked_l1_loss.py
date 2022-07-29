import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss.mean(2)
        loss *= mask
        return loss.sum() / mask.sum()
