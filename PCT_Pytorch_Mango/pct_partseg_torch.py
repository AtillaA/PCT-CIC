import torch 
from torch import nn 
from torch import cat as concat 
import torch.nn.functional as F
import numpy as np 
import math 
from util import sample_and_group 

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Point_Transformer_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(Point_Transformer_partseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.pos_xyz = nn.Conv1d(3, 128, 1)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_intermediate = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn_int = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        # xyz = x
        # print(N)
        batch_size, _, N = x.size()
        xyz = x.permute(0, 2, 1)
        # Grey: Input Embedding
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        #Include sample and grouping
        x = x.permute(0, 2, 1)
        # print(x_sampling.shape)
        # print(xyz.shape, x.shape)
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.15, nsample=32, xyz=xyz, points=x)         
        # print(f'new feature after sample_and_group1: {new_feature.shape}')
        feature_0 = self.gather_local_0(new_feature)
        # print(f'Feature after gather0: {feature_0.shape}')
        feature = feature_0.permute(0, 2, 1)
        # print(f'Feature after permute: {feature.shape}')
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        # print(f'new feature after sample_and_group2: {new_feature.shape}')
        feature_1 = self.gather_local_1(new_feature)

        # Yellow attention blocks

        new_xyz = new_xyz.permute(0, 2, 1)
        # print('finished gathering')
        xyz_mod = self.pos_xyz(new_xyz)
        # print('working after posxyz')
        x_pt = self.relu(self.bn_int(self.conv_intermediate(feature_1)))

        # print(x_pt.shape, xyz_mod.shape)
        
        x1 = self.sa1(x_pt + xyz_mod)
        x2 = self.sa2(x1 + xyz_mod)
        x3 = self.sa3(x2 + xyz_mod)
        x4 = self.sa4(x3 + xyz_mod)
        
        # Concat
        x = concat((x1, x2, x3, x4), dim=1)
        
        # Linear layer
        x = self.conv_fuse(x)
        
        x_max, x_max_indices = torch.max(x, 2)
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        # MA-Pool: Global feature from max and avg as in page 4
        x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
        x = concat((x, x_global_feature), 1) # 1024 * 3 + 64 
        
        # segmentation starts

        # LBRD (dark green)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        # LBR (green)
        x = self.relu(self.bns2(self.convs2(x)))
        # Linear (light green)
        x = self.convs3(x)
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        #self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x) # b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
