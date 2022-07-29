import torch.nn as nn
import torch.nn.functional as F


class MaskedBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedBinaryCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        loss = F.binary_cross_entropy(logit, target, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()
