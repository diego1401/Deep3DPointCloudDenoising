import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedGlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalAvgPool1d, self).__init__()

    def forward(self, mask, features):
        out = features.sum(-1)
        pcl_num = mask.sum(-1)
        out /= pcl_num[:, None]
        return out


class DiscriminatorHead(nn.Module):
    def __init__(self, width):
        """ The head for our discriminator block

        Returns:
            logits: (B, 1)
        """
        super(DiscriminatorHead, self).__init__()
        self.num_classes = 1
        self.pool = MaskedGlobalAvgPool1d()
        self.classifier = nn.Sequential(
            nn.Linear(16 * width, 8 * width),
            nn.BatchNorm1d(8 * width),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8 * width, 4 * width),
            nn.BatchNorm1d(4 * width),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4 * width, 2 * width),
            nn.BatchNorm1d(2 * width),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2 * width, self.num_classes),
            nn.Sigmoid())

    def forward(self, end_points):
        pooled_features = self.pool(end_points['res5_mask'], end_points['res5_features'])
        return self.classifier(pooled_features)
