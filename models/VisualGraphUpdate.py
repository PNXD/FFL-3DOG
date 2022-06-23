import numpy as np
import torch
import torch.nn as nn
import utils.ops as ops
import sys
import os

from utils.numerical_stability_softmax import numerical_stability_masked_softmax

sys.path.append(os.path.join(os.getcwd()))


class VisualSceneGraphV1(nn.Module):

    """
    Message propagation by using the structure information.
    Guided by the language structure.
    In this version, we just discard the edge information, and the message just in the nodes.
    """

    def __init__(self, visual_dim=128):
        super(VisualSceneGraphV1, self).__init__()
        self.visual_dim = visual_dim

        #  inter message propagation
        self.rel_update_embedding = ops.Linear(self.visual_dim * 3, self.visual_dim)
        self.visual_joint_trans_sub = ops.Linear(2 * self.visual_dim, self.visual_dim)
        self.visual_joint_trans_obj = ops.Linear(2 * self.visual_dim, self.visual_dim)

        self.visual_ctx_embedding = ops.Linear(2 * self.visual_dim, self.visual_dim)

    def forward(self, visual_feat=None, rel_visual_feat=None, conn_map=None, topN_boxes_scores=None, target_id=None):

        num_phrase, topN = topN_boxes_scores.shape
        conn_map_numpy = conn_map.detach().cpu().numpy()
        sub_ind, obj_ind = np.where(conn_map_numpy >= 0)

        update_nodes = []
        no_update_nodes = []
        for pid in range(num_phrase):
            if pid == target_id:
                inv_id = np.arange(topN) + pid*topN
                update_nodes.append(inv_id)
            else:
                non_inv_id = np.arange(topN) + pid*topN
                no_update_nodes.append(non_inv_id)

        no_update_nodes.append(np.array([]))
        no_update_nodes.append(np.array([]))
        no_update_nodes = np.concatenate(tuple(no_update_nodes))
        update_nodes = np.concatenate(tuple(update_nodes))

        """ aggregate the node information """
        visual_feat_sub = visual_feat[sub_ind]
        visual_feat_obj = visual_feat[obj_ind]
        updated_rel_feat = self.rel_update_embedding(
            torch.cat((visual_feat_sub, visual_feat_obj, rel_visual_feat), 1))

        visual_trans_sub = self.visual_joint_trans_sub(torch.cat((visual_feat_sub, updated_rel_feat), 1))
        visual_trans_obj = self.visual_joint_trans_obj(torch.cat((visual_feat_obj, updated_rel_feat), 1))

        weight_atten = torch.zeros(conn_map.shape).cuda()
        weight_atten[sub_ind, obj_ind] = (visual_trans_sub*visual_trans_obj).sum(1)/(self.visual_dim**0.5)

        weight_sub = numerical_stability_masked_softmax(vec=weight_atten.float(), mask=conn_map.ge(0), dim=1,
                                                        num_phrases=num_phrase,
                                                        topN=topN)  # softmax along the dim1
        weight_obj = numerical_stability_masked_softmax(vec=weight_atten.float(), mask=conn_map.ge(0), dim=0,
                                                        num_phrases=num_phrase,
                                                        topN=topN)  # softmax along the dim0

        """ aggregate the relation information into subject context node and object context node """
        visual_joint_sub = visual_feat.unsqueeze(1).repeat(1, topN*num_phrase, 1)
        visual_joint_obj = visual_feat.unsqueeze(0).repeat(topN*num_phrase, 1, 1)

        visual_rel = torch.zeros(conn_map.shape[0], conn_map.shape[1], self.visual_dim).cuda()
        visual_rel[sub_ind, obj_ind] = updated_rel_feat
        visual_joint_sub = torch.cat((visual_joint_sub, visual_rel), 2)
        visual_joint_obj = torch.cat((visual_joint_obj, visual_rel), 2)

        visual_joint = (visual_joint_obj * weight_sub.unsqueeze(2)).sum(1) + (visual_joint_sub * weight_obj.unsqueeze(2)).sum(0)
        update_visual_feat = torch.zeros_like(visual_feat).cuda()

        update_visual_feat[update_nodes] = visual_feat[update_nodes] + self.visual_ctx_embedding(visual_joint[update_nodes])

        update_visual_feat[no_update_nodes] = visual_feat[no_update_nodes]

        return updated_rel_feat, update_visual_feat
