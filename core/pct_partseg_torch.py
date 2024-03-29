from platform import architecture
import torch 
from torch import nn 
from torch import cat as concat 

from curvenet_util import CIC

curve_config = {
        'default': [[100, 5], [100, 5], None, None, None]
    }


class Point_Transformer_partseg(nn.Module):
    def __init__(self, part_num=50, model_type='cic4_pos'):
        super(Point_Transformer_partseg, self).__init__()

        self.model_type = model_type
        k = 32
        additional_channel = 32
        setting='default'

        self.part_num = part_num

        # ------------------------------------- PCT ------------------------------------- #
        if model_type == 'pct':
            self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
        # ------------------------------------------------------------------------------- #
        # ------------------------------------- CIC ------------------------------------- #
        else:
            self.conv1 = nn.Conv1d(3, 16, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=1, bias=False)

            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(32)

            if model_type == 'cic2_pos':
                self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel, output_channels=128, bottleneck_ratio=2, curve_config=curve_config[setting][0])
                self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, curve_config=curve_config[setting][0])
            
            elif model_type == 'cic4' or model_type == 'cic4_pos':
                self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, curve_config=curve_config[setting][0])
                self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, curve_config=curve_config[setting][0])

                self.cic21 = CIC(npoint=2048, radius=0.4, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, curve_config=curve_config[setting][1])
                self.cic22 = CIC(npoint=2048, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, curve_config=curve_config[setting][1])
        # ------------------------------------------------------------------------------- #

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

        if model_type == 'pct' or model_type == 'cic4':
            self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        elif model_type == 'cic2_pos' or model_type == 'cic4_pos':
            self.convs1 = nn.Conv1d(1024 * 3 + 64 + 3, 512, 1)


        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):

        batch_size, _, N = x.size()
        xyz = x
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))


        if self.model_type == 'pct':
            x1 = self.sa1(x)
        elif self.model_type == 'cic2_pos':
            l1_xyz, l1_points = self.cic11(xyz, x)
            l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
            x1 = self.sa1(l1_points)
        elif self.model_type == 'cic4' or self.model_type == 'cic4_pos':
            l1_xyz, l1_points = self.cic11(xyz, x)
            l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
            l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
            l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)
            x1 = self.sa1(l2_points)
        
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)

        x_max, x_max_indices = torch.max(x, 2)
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        if self.model_type == 'pct' or self.model_type == 'cic4':
            x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64 
        elif self.model_type == 'cic2_pos':
            x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature, l1_xyz), 1) # 1024 + 64 
        elif self.model_type == 'cic4_pos':
            x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature, l2_xyz), 1) # 1024 + 64 

        x = concat((x, x_global_feature), 1) # 1024 * 3 + 64 
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
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
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
