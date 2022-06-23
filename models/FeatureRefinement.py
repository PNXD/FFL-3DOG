import torch
from torch import nn
import numpy as np
import os
import sys
from utils.numerical_stability_softmax import numerical_stability_masked_softmax
import utils.ops as ops
from config.config import CONF

sys.path.append(os.path.join(os.getcwd()))


class LanguageSceneGraphV1(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(LanguageSceneGraphV1, self).__init__()
        self.hidden_dim = hidden_dim

        # Update edges
        self.rel_embed = ops.Linear(3 * self.hidden_dim, self.hidden_dim)

        # Generate attention weights
        self.trans_sub = ops.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.trans_obj = ops.Linear(2 * self.hidden_dim, self.hidden_dim)

        # Update the nodes involved
        self.phr_embed = ops.Linear(2*self.hidden_dim, self.hidden_dim)

    def forward(self, phrase_feat, rel_feat, rel_conn_mat, target_id, eps=1e-6):
        """
        :param phrase_feats: feature vectors
        :param rel_feat:
        :param rel_conn_mat: (2, connections)
        :param target_id:
        :return updated feature vectors
        """

        num_phrase = phrase_feat.shape[0]
        subject_nodes = rel_conn_mat[0]
        object_nodes = rel_conn_mat[1]

        # Update relation feature
        updated_rel_feat = self.rel_embed(torch.cat((phrase_feat[subject_nodes], phrase_feat[object_nodes], rel_feat), 1))

        phr_conn_mat = np.zeros((num_phrase, num_phrase))
        subject_nodes = subject_nodes.detach().cpu().numpy()
        object_nodes = object_nodes.detach().cpu().numpy()
        phr_conn_mat[subject_nodes, object_nodes] = 1
        phr_conn_mat = torch.FloatTensor(phr_conn_mat).cuda()

        trans_sub = self.trans_sub(torch.cat([phrase_feat[subject_nodes], updated_rel_feat], 1))
        trans_obj = self.trans_obj(torch.cat([phrase_feat[object_nodes], updated_rel_feat], 1))

        atte = (trans_sub * trans_obj).sum(1)/(trans_sub.shape[1]**0.5)
        atte_map = torch.zeros(num_phrase, num_phrase).cuda()
        atte_map[subject_nodes, object_nodes] = atte

        # Attention weight
        atte_sub = numerical_stability_masked_softmax(vec=atte_map.float(), mask=phr_conn_mat, dim=1)
        atte_obj = numerical_stability_masked_softmax(vec=atte_map.float(), mask=phr_conn_mat, dim=0)

        feature_4_sub = phrase_feat.unsqueeze(0).repeat(num_phrase, 1, 1)
        feature_4_obj = phrase_feat.unsqueeze(1).repeat(1, num_phrase, 1)

        rel_feature_mat = torch.zeros(num_phrase, num_phrase, phrase_feat.shape[1]).cuda()
        rel_feature_mat[subject_nodes, object_nodes] = updated_rel_feat
        feature_4_sub = torch.cat([feature_4_sub, rel_feature_mat], 2)
        feature_4_obj = torch.cat([feature_4_obj, rel_feature_mat], 2)

        phr_context_feat = (feature_4_sub * atte_sub.unsqueeze(2)).sum(1) + (feature_4_obj * atte_obj.unsqueeze(2)).sum(0)

        # Identify the nodes that need to be updated
        update_nodes = np.array(target_id)
        no_update_nodes = []
        for i in range(num_phrase):
            if i != target_id:
                no_update_nodes.append(i)
        no_update_nodes = np.array(no_update_nodes)

        # Update phrase feature
        update_phrase_feat_unified = torch.zeros(phrase_feat.shape[0], phrase_feat.shape[1]).cuda()

        update_phrase_feat_unified[update_nodes] = phrase_feat[update_nodes] + self.phr_embed(phr_context_feat[update_nodes])

        update_phrase_feat_unified[no_update_nodes] = phrase_feat[no_update_nodes]

        return update_phrase_feat_unified, updated_rel_feat


if __name__ == '__main__':

    import torch
