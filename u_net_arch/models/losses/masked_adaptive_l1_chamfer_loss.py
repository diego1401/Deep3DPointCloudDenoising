from distutils.command import clean
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from .chamfer_distance_aux import chamfer_distance


class MaskedAdaptiveL1ChamferLoss(nn.Module):
    def __init__(self,converging_to):
        super(MaskedAdaptiveL1ChamferLoss, self).__init__()
        self.converging_to = converging_to

    def forward(self, pred, target, mask, points):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss.mean(2)
        loss *= mask
        l1 = loss.sum() / mask.sum()

        clean_points = points + target
        pred_points = points + pred

        # Option 1

        cd = 0
        number_batches = mask.shape[0]
        for bi in range(number_batches):
            mask_i = mask[bi].type(torch.bool)
            
            clean_points_bi = clean_points[bi,mask_i,:].unsqueeze(0)
            pred_points_bi = pred_points[bi,mask_i,:].unsqueeze(0)
            loss2, _ = chamfer_distance(clean_points_bi, pred_points_bi,
                         batch_reduction="sum",point_reduction='mean',norm_type="L1") # We need it to be comparable to the l1 loss
        
            cd += loss2
        
        cd /= number_batches

        # Option 2

        # clean_points *= mask.unsqueeze(2)
        # pred_points *= mask.unsqueeze(2)

        # loss2, _ = chamfer_distance(clean_points, pred_points,batch_reduction=None,point_reduction='sum')
        # # The divion by mask.sum() serves as the mean.
        # l2 = loss2.sum()/mask.sum()

        if self.converging_to == 'chamfer':
            return l1 + torch.exp(-l1) * cd
        elif self.converging_to == 'L1':
            return cd + torch.exp(-cd) *l1
        else:
            raise ValueError(f"Limit of loss {self.converging_to} not implemented")
 