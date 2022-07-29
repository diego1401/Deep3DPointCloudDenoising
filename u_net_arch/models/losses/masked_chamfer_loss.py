from distutils.command import clean
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from .chamfer_distance_aux import chamfer_distance


class MaskedChamferLoss(nn.Module):
    def __init__(self,norm_type="L2"):
        super(MaskedChamferLoss, self).__init__()
        self.norm_type = norm_type

    def forward(self, pred, target, mask, points):
        clean_points = points + target
        pred_points = points + pred

        cd = 0
        for bi in range(mask.shape[0]):
            mask_i = mask[bi].type(torch.bool)
            
            clean_points_bi = clean_points[bi,mask_i,:].unsqueeze(0)
            pred_points_bi = pred_points[bi,mask_i,:].unsqueeze(0)
            loss2, _ = chamfer_distance(clean_points_bi, pred_points_bi,batch_reduction="sum",
                                        point_reduction='mean',norm_type=self.norm_type)
            cd += loss2
        number_batches = mask.shape[0]
        return cd/number_batches