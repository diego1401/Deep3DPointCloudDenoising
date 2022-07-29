import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedOffsetLoss(nn.Module):
    def __init__(self):
        super(MaskedOffsetLoss, self).__init__()

    def forward(self, pred, target, mask):
        weight = 1/torch.norm(target,dim=2).unsqueeze(2)
        weight = torch.clip(weight,1e-6,2) # Could be irrelevant up to 2 times as important as unweighed
        loss = F.l1_loss(pred, target, reduction='none')  * weight
        loss = loss.mean(2)
        loss *= mask
        return loss.sum() / mask.sum()
