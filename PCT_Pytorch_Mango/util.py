import torch
import torch.nn.functional as F
import time
import torch_geometric.nn as geo
# from pointnet2_ops import pointnet2_utils

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # print(f'Source shape {src.shape}')
    # print(f'dst shape {dst.shape}')
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    # print(nsample, xyz.shape, new_xyz.shape)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)

    return group_idx

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def fps(xyz, M): # ours
    """
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)

    start_for = time.time()

    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]

    end_for = time.time()
    print(f'One for pass: {end_for - start_for}')

    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()

    # print(f'XYZ in sample and group {xyz.shape}')

    # fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    
    start_fps = time.time()
    fps_idx = fps(xyz, npoint).long()
    # fps_idx = geo.fps(xyz, None, 1.0, False)
    end_fps = time.time()
    print(f'One fps pass: {end_fps - start_fps}')

    # trial_fps = fps(xyz, npoint)
    # print("trial_fps: ",trial_fps.shape)
    # print(trial_fps[0])
    # print("-"*10)
    # print(fps_idx[0])
    # print("geo.fps: ",fps_idx.shape)


    new_xyz = index_points(xyz, fps_idx)
    # print(f'new xyz shape {new_xyz.shape}')
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # not used? try to remove and run?
    grouped_points = index_points(points, idx)
    # Matrix sub
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    
    # Concat & Repeat
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

"""

class CIC(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, output_channels, bottleneck_ratio=2, mlp_num=2, curve_config=None):
        super(CIC, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(planes, k, curve_config[0], curve_config[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,
                      planes,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels))

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm1d(output_channels))

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.maxpool = MaskedMaxPool(npoint, radius, k)

        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz, x):
 
        # max pool
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(
                xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)  # bs, c', n

        idx = knn(xyz, self.k)

        if self.use_curve:
            # curve grouping
            curves = self.curvegrouping(x, xyz, idx[:,:,1:]) # avoid self-loop

            # curve aggregation
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:,:,:self.k]) #bs, c', n, k

        x = self.conv2(x)  # bs, c, n

        if self.in_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)

        return xyz, x

class CurveAggregation(nn.Module):
    def __init__(self, in_channel):
        super(CurveAggregation, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convb = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convc = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convn = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convl = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2,
                      in_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channel))
        self.line_conv_att = nn.Conv2d(in_channel,
                                       1,
                                       kernel_size=1,
                                       bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l

        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  #bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  #bs, c, c_l

        curver_inter = self.conva(curver_inter) # bs, mid, n
        curves_intra = self.convb(curves_intra) # bs, mid ,n

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1) # bs, n, c_n
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1) # bs, l, c_l
        

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)

        curve_features = torch.cat((x_inter, x_intra),dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)



#CurveNet Grouping and aggregation

class CurveGrouping(nn.Module):
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(CurveGrouping, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.in_channel = in_channel
        self.k = k

        self.att = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)

        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x, xyz, idx):
        # starting point selection in self attention style
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att

        _, start_index = torch.topk(x_att,
                                    self.curve_num,
                                    dim=2,
                                    sorted=False)
        start_index = start_index.squeeze().unsqueeze(2)

        curves = self.walk(xyz, x, idx, start_index)  #bs, c, c_n, c_l
        
        return curves

class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, k):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz, features):
        sub_xyz, neighborhood_features = sample_and_group(self.npoint, self.radius, self.k, xyz, features.transpose(1,2))

        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # bs, c, n, 1
        sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return sub_xyz, sub_features

"""