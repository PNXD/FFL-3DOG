import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from models.pointnet2 import pytorch_utils
from config.config import CONF

sys.path.append(os.path.join(os.getcwd()))


class SceneBoxEmb(nn.Module):
    def __init__(self, input_channels, num_proposal):
        super(SceneBoxEmb, self).__init__()
        self.in_dim = input_channels + 3
        self.output_size = CONF.MODEL.SceneBoxEmb.OUT_DIM
        self.pooling = 'max'
        self.num_proposal = num_proposal

        mlp_spec = [self.in_dim, 64, 128, 256]
        self.mlp_module = pytorch_utils.SharedMLP(mlp_spec, bn=True)
        self.gs_conv1 = torch.nn.Conv1d(512, self.output_size, 1)

    def forward(self, union_box, box_features, agg_xyz, seed_feature, seed_xyz, box_feature_union):
        agg_f = torch.zeros(union_box[0].shape[0], self.num_proposal, box_features.shape[-1]).cuda().half()

        union_box = union_box[0]
        num = 0
        if 650 < union_box.shape[0] <= 750:
            num = num+1
        bomin = union_box[:, :3] - 1 / 2 * union_box[:, 3:6]
        bomax = union_box[:, :3] + 1 / 2 * union_box[:, 3:6]
        bo1 = torch.ge(torch.unsqueeze(agg_xyz, 0).repeat(union_box.shape[0], 1, 1)[:, :, :3], torch.unsqueeze(bomin, 1), out=None)
        bo2 = torch.ge(torch.unsqueeze(bomax, 1), torch.unsqueeze(agg_xyz, 0).repeat(union_box.shape[0], 1, 1)[:, :, :3], out=None)
        bo = torch.cat((bo1, bo2), 2)
        bo = torch.sum(bo, 2, out=None)
        ids = torch.squeeze(torch.nonzero(bo > 5), 1)
        agg_f[ids[:, 0], ids[:, 1], :] = box_features[ids[:, 1], :].half()  # num_nms*num_id*128
        g_features_2 = F.max_pool1d(agg_f.permute(0, 2, 1), kernel_size=agg_f.size(1)).half()  # num_nms*128*1

        po1 = torch.ge(torch.unsqueeze(seed_xyz, 0).repeat(union_box.shape[0], 1, 1)[:, :, :3], torch.unsqueeze(bomin, 1), out=None)
        po2 = torch.ge(torch.unsqueeze(bomax, 1), torch.unsqueeze(seed_xyz, 0).repeat(union_box.shape[0], 1, 1)[:, :, :3], out=None)
        po = torch.cat((po1, po2), 2)
        po = torch.sum(po, 2, out=None)
        if union_box.shape[0] > 750 or num >= 3:
            g_features_1 = torch.zeros(union_box.shape[0], seed_feature.shape[0], 1).cuda().half()
            for i in range(po.shape[0]):
                idx = torch.squeeze(torch.nonzero(po[i, :] > 5), 1)
                if idx.shape[0] != 0:
                    seed_f = seed_feature[:, idx].half()
                else:
                    seed_f = torch.zeros(seed_feature.shape[0], 1).cuda().half()
                global_feat1 = F.max_pool1d(torch.unsqueeze(seed_f, 0), kernel_size=seed_f.size(1)).half()  # num_nms*256*1
                g_features_1[i] = global_feat1
        else:
            vote_f = torch.zeros(union_box.shape[0], seed_xyz.shape[0], seed_feature.shape[0]).cuda().half()
            idx = torch.squeeze(torch.nonzero(po > 5), 1)
            vote_f[idx[:, 0], idx[:, 1], :] = seed_feature.transpose(1, 0)[idx[:, 1], :].half()
            g_features_1 = F.max_pool1d(vote_f.permute(0, 2, 1), kernel_size=vote_f.size(1)).half()  # num_nms*256*1

        global_feat = torch.cat((g_features_1, g_features_2), 1).float()  # num_nms*(256+128)*1
        global_features = torch.cat((global_feat, box_feature_union.permute(0, 2, 1)), 1)  # num_nms*512*1
        global_features = self.gs_conv1(global_features)
        global_features = torch.sigmoid(torch.log(torch.abs(global_features+1e-6)))  # num_nms*128*1
        global_features = torch.squeeze(global_features, -1)  # num_nms*128
        return global_features
