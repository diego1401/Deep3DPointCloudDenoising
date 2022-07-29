import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pt_utils import MaskedUpsample



class MultiDimHeadResNet(nn.Module):
    def __init__(self, num_classes,width, base_radius, nsamples,isGAN=False):
        """An Multi Dimensional head for ResNet backbone.
        E.g Offset regresion -> dim 3
            Outlier detection -> dim 1 (or 2 if hot vector)

        Args:
            num_classes: class num.
            width: the base channel num.
            base_radius: the base ball query radius.
            nsamples: neighborhood limits for each layer, a List of int.

        Returns:
            logits: (B, num_classes, N)
        """
        super(MultiDimHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        head_layers = [nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(width // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 2,self.num_classes , kernel_size=1, bias=True),
                                #   nn.Tanh()
                                  ]
                                  
        self.head = nn.Sequential(*head_layers)
        

    def forward(self, end_points):
        features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],
                            end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])
        features = torch.cat([features, end_points['res4_features']], 1)
        features = self.up_conv0(features)

        features = self.up1(end_points['res3_xyz'], end_points['res4_xyz'],
                            end_points['res3_mask'], end_points['res4_mask'], features)
        features = torch.cat([features, end_points['res3_features']], 1)
        features = self.up_conv1(features)

        features = self.up2(end_points['res2_xyz'], end_points['res3_xyz'],
                            end_points['res2_mask'], end_points['res3_mask'], features)
        features = torch.cat([features, end_points['res2_features']], 1)
        features = self.up_conv2(features)

        features = self.up3(end_points['res1_xyz'], end_points['res2_xyz'],
                            end_points['res1_mask'], end_points['res2_mask'], features)
        features = torch.cat([features, end_points['res1_features']], 1)
        features = self.up_conv3(features)

        offset = self.head(features)

        return offset

