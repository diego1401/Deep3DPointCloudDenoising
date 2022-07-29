import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedOutlierLoss(nn.Module):
    def __init__(self):
        super(MaskedOutlierLoss, self).__init__()

    def forward(self, logit, target, true_offsets, mask):
        loss = F.binary_cross_entropy(logit, target, reduction='none') * torch.norm(true_offsets,dim=2)
        loss *= mask
        return loss.sum() / mask.sum()
