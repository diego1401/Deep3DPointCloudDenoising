import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from pt_utils import MaskedQueryAndGroup,MaskedNearestQueryAndGroup

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def transformer_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size*num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    position_vector = (x - neighbor_x).permute(0, 3, 1, 2).contiguous() # B,3,N,k

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size*num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    neighbor_feat = neighbor_feat.permute(0, 3, 1, 2).contiguous() # B,C,N,k

    return position_vector, neighbor_feat


class OffsetAttention(nn.Module):
    def __init__(self, channels, ratio = 8):
        super(OffsetAttention, self).__init__()

        self.bn1 = nn.BatchNorm1d(channels // ratio)
        self.bn2 = nn.BatchNorm1d(channels // ratio)
        self.bn3 = nn.BatchNorm1d(channels)

        self.q_conv = nn.Conv1d(channels, channels // ratio, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // ratio, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.act(self.bn1(self.q_conv(x))).permute(0, 2, 1)  # b, n, c/ratio
        x_k = self.act(self.bn2(self.k_conv(x)))  # b, c/ratio, n
        x_v = self.act(self.bn3(self.v_conv(x)))  # b, c, n
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x



class PointAttentionNetwork(nn.Module):
    def __init__(self,C, ratio = 8):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(C//ratio)
        self.bn2 = nn.BatchNorm1d(C//ratio)
        self.bn3 = nn.BatchNorm1d(C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b,c,n = x.shape

        a = self.conv1(x).permute(0,2,1) # b, n, c/ratio

        b = self.conv2(x) # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b)) # b,n,n

        d = self.conv3(x) # b,c,n
        out = x + torch.bmm(d, s.permute(0, 2, 1))

        return out


class ShapeContext(nn.Module):
    def __init__(self, C, ratio=8):
        super(ShapeContext, self).__init__()
        self.bn1 = nn.BatchNorm1d(C//ratio)
        self.bn2 = nn.BatchNorm1d(C//ratio)
        self.bn3 = nn.BatchNorm1d(C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b,c,n = x.shape

        q = self.conv1(x).permute(0, 2, 1) #b, n, c/ratio

        k = self.conv2(x) #b, c/ratio, n

        a = self.softmax(torch.bmm(q, k)) # b, n, n

        v = self.conv3(x) #b, c, n

        out = torch.bmm(v, a.permute(0, 2, 1))  #b, c, n

        return out + v


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):

    def __init__(self, in_dim, ratio = 8):
        super(CrissCrossAttention,self).__init__()

        self.bn1 = nn.BatchNorm2d(in_dim//ratio)
        self.bn2 = nn.BatchNorm2d(in_dim//ratio)
        self.bn3 = nn.BatchNorm2d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, data):

        b, c, n = data.shape
        x = data.view(b,c,n,1) #b, c, n, 1
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x) # b, c/ratio, n, 1
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) #b*1, n, c/ratio
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)# b*n, 1, c/ratio

        proj_key = self.key_conv(x) #b, c/ratio, n, 1
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) # b*1, c/ratio, n
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) # b*n, c/ratio, 1


        proj_value = self.value_conv(x) # b, c, n, 1
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) # b*1, c, n
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) # b*n, c, 1

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3) # b, n, 1, n
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) # b, n, 1, 1


        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) # b, n, 1, 1+n
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) # b, n, n

        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) # b*n, 1, 1

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1).contiguous() # b, c, n, 1
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3).contiguous() # b, c, n, 1
        # print(out_H.size(),out_W.size())
        out = self.gamma*(out_H + out_W) + x # b, c, n ,1
        return out.view(b, c, n).contiguous()


class PAM(nn.Module):
    def __init__(self, C):
        super(PAM, self).__init__()
        self.dim = C
        self.conv1 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels = C, out_channels=C, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        b, c, n =  x.shape

        out1 = self.conv1(x).view(b, -1, n).permute(0, 2, 1) # b, n, c/latent

        out2 = self.conv2(x).view(b, -1, n) # b,c/latent,n

        attention_matrix = self.softmax(torch.bmm(out1, out2)) # b,n,n

        out3 = self.conv3(x).view(b, -1, n) # b,c,n

        attention = torch.bmm(out3, attention_matrix.permute(0, 2, 1))

        out = self.gamma * attention.view(b, c, n) + x
        return  out


class CAM(nn.Module):
    def __init__(self, C):
        super(CAM, self).__init__()
        self.dim = C
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, n = x.shape

        out1 = x.view(b, c, -1)  # b,c,n
        out2 = x.view(b, c, -1).permute(0, 2, 1) # b,n,c
        attention_matrix = torch.bmm(out1, out2) # b,c,c
        attention_matrix = self.softmax(torch.max(attention_matrix, -1, keepdim=True)[0].expand_as(attention_matrix) - attention_matrix) # b,c,c

        out3 = x.view(b, c, -1) # b,c,n

        out = torch.bmm(attention_matrix, out3) # b,c,n
        out = self.gamma * out.view(b, c, n) + x

        return out

class DualAttention(nn.Module):
    def __init__(self,C):
        super(DualAttention,self).__init__()

        self.dim = C

        self.cam = CAM(C)
        self.pam = PAM(C)

    def forward(self,x):
        return self.cam(x)+self.pam(x)


class ChannelAttentionModule(nn.Module):
    """ this function is used to achieve the channel attention module in CBAM paper"""
    def __init__(self, C, ratio=8): # getting from the CBAM paper, ratio=16
        super(ChannelAttentionModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels= C // ratio, out_channels=C, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        out1 = torch.mean(x, dim=-1, keepdim=True)  # b, c, 1
        out1 = self.mlp(out1) # b, c, 1

        out2 = nn.AdaptiveMaxPool1d(1)(x) # b, c, 1
        out2 = self.mlp(out2) # b, c, 1

        out = self.sigmoid(out1 + out2)

        return out * x


class SpatialAttentionModule(nn.Module):
    """ this function is used to achieve the spatial attention module in CBAM paper"""
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = torch.mean(x,dim=1,keepdim=True) #B,1,N

        out2, _ = torch.max(x, dim=1,keepdim=True)#B,1,N

        out = torch.cat([out2, out1], dim=1) #B,2,N

        out = self.conv1(out) #B,1,N
        out = self.bn(out) #B,1,N
        out =self.relu(out) #B,1,N

        out = self.sigmoid(out) #b, c, n
        return out * x

class CBAMAttention(nn.Module):
    def __init__(self,C):
        super(CBAMAttention,self).__init__()

        self.dim = C

        self.cbam_ca = ChannelAttentionModule(C)
        self.cbam_sa = SpatialAttentionModule()

    def forward(self,x):
        x = self.cbam_ca(x)
        x = self.cbam_sa(x)
        return x

class Point_Transformer(nn.Module):
    def __init__(self, input_features_dim,radius,nsample):
        super(Point_Transformer, self).__init__()

        self.conv_theta1 = nn.Conv2d(3, input_features_dim, 1)
        self.conv_theta2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_theta = nn.BatchNorm2d(input_features_dim)

        self.conv_phi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_psi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_alpha = nn.Conv2d(input_features_dim, input_features_dim, 1)

        self.conv_gamma1 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_gamma2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_gamma = nn.BatchNorm2d(input_features_dim)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)
        self.grouper_nn = MaskedNearestQueryAndGroup(use_xyz=False, ret_grouped_xyz=True)

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):

        # position_vector, x_j = transformer_neighbors(xyz, features, k=k)
        x_j, position_vector, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask, support_mask, support_features)
        x_i, _,_ = self.grouper_nn(query_xyz, support_xyz, query_mask, support_mask, support_features)
        x_i = x_i[...,0]

        delta = F.relu(self.bn_conv_theta(self.conv_theta2(self.conv_theta1(position_vector)))) # B,C,N,k
        # corrections for x_i = nearest neighbor unsqueezed
        x_i = torch.unsqueeze(x_i, dim=-1).repeat(1, 1, 1, x_j.shape[-1]) # B,C,N,k

        linear_x_i = self.conv_phi(x_i) # B,C,N,k

        linear_x_j = self.conv_psi(x_j) # B,C,N,k

        relation_x = linear_x_i - linear_x_j + delta # B,C,N,k
        relation_x = F.relu(self.bn_conv_gamma(self.conv_gamma2(self.conv_gamma1(relation_x)))) # B,C,N,k

        weights = F.softmax(relation_x, dim=-1) # B,C,N,k
        features = self.conv_alpha(x_j) + delta # B,C,N,k

        feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
        feature_mask = feature_mask[:, None, :, :]
        features *= feature_mask
        f_out = weights * features # B,C,N,k
        f_out = torch.sum(f_out, dim=-1) # B,C,N

        return f_out


class NonLocalModule(nn.Module):
    def __init__(self, C, latent= 8):
        super(NonLocalModule, self).__init__()
        self.inputChannel = C
        self.latentChannel = C // latent

        self.bn1 = nn.BatchNorm1d(C//latent)
        self.bn2 = nn.BatchNorm1d(C//latent)
        self.bn3 = nn.BatchNorm1d(C//latent)
        self.bn4 = nn.BatchNorm1d(C)

        self.cov1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//latent, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.cov2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//latent, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.cov3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//latent, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv1d(in_channels=C//latent, out_channels=C, kernel_size=1, bias=False),
                                self.bn4,
                                nn.ReLU())

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, n = x.shape

        out1 = self.cov1(x).view(b, -1, n).permute(0, 2, 1) #b,n,c/latent
        out2 = self.cov2(x).view(b, -1, n) #b, c/latent, n

        attention_matrix = self.softmax(torch.bmm(out1, out2)) # b,n,n

        out3 = self.cov3(x).view(b, -1, n) # b,c/latent,n

        attention = torch.bmm(out3, attention_matrix.permute(0, 2, 1)) # b,c/latent,n

        out = self.out_conv(attention) #b,c,n

        return self.gamma*out + x


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""
    def __init__(self, in_dim, in_pts):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_pts//8)
        self.bn2 = nn.BatchNorm1d(in_pts//8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=in_pts, out_channels=in_pts//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=in_pts, out_channels=in_pts//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha*out + x
        return out


class SE(nn.Module):
    def __init__(self, C, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(C, C // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // r, C, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, C, _ = x.shape
        out = self.squeeze(x).view(b, C)
        out = self.excitation(out).view(b, C, 1)
        return x * out.expand_as(x)
