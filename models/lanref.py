import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models.phrase_embedding import PhraseEmbeddingSent
from models.sceneboxemb import SceneBoxEmb
from config.config import CONF
from models.FeatureRefinement import LanguageSceneGraphV1
from models.VisualGraphUpdate import VisualSceneGraphV1
from utils.generate_union_region import generate_union_region_boxes
from utils.bounding_box import BoxList
from utils.box_coder import BoxCoder

sys.path.append(os.path.join(os.getcwd()))

MAX_NUM_P = 25


class lanref(nn.Module):
    def __init__(self, input_channels, use_bidir, use_lang_classifier, num_proposal, hidden_size=128):
        super(lanref, self).__init__()

        self.recognition_dim = CONF.LANREF.recognition_dim
        self.phrase_embed_dim = CONF.LANREF.Phrase_embed_dim
        self.hidden_size = hidden_size

        self.use_bidir = use_bidir
        self.use_lang_classifier = use_lang_classifier
        self.input_channels = input_channels

        # Language Encoding
        self.phrase_embed = PhraseEmbeddingSent(self.phrase_embed_dim, use_bidir)

        # Union Box Feature Learning
        self.getboxfeature = SceneBoxEmb(self.input_channels, num_proposal)

        # Language Scene Graph
        self.phrase_mps = LanguageSceneGraphV1(hidden_dim=self.phrase_embed_dim)

        # Visual Scene Graph
        self.visual_graph = VisualSceneGraphV1()

        bbox_reg_weights = CONF.MODEL.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

        # Feature Similarity
        self.similarity_input_dim = self.recognition_dim + self.phrase_embed_dim*3
        self.similarity = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self.box_reg = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6)
        )
        self.similarity_topN = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        self.box_reg_topN = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, data_dict):
        """

        """

        pred_boxes = data_dict["precomp_boxes"]
        batch_size = pred_boxes.shape[0]
        objectness_masks = data_dict["objectness_scores"].max(2)[1].float().unsqueeze(2)

        aggregated_features = data_dict["aggregated_vote_features"]
        aggregated_xyz = data_dict["aggregated_vote_xyz"]

        seed_features = data_dict["seed_features"]
        seed_xyz = data_dict["seed_xyz"]

        """ Language encoding """
        batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_target_id = self.phrase_embed(data_dict)

        batch_final_box_det = []

        batch_pred_similarity = []
        batch_reg_offset = []

        batch_topN_boxes = []
        batch_pred_similarity_topN = []
        batch_reg_offset_topN = []

        batch_target_embed = []
        batch_topN_target_ids = []

        batch_pred_scores = []

        for bid in range(batch_size):
            pred_boxes_i = pred_boxes[bid]
            target_id = batch_target_id[bid]

            box_features = aggregated_features[bid]
            phrase_embed_i = batch_phrase_embed[bid]

            num_box = pred_boxes_i.size(0)
            num_phrase = phrase_embed_i.size(0)

            all_phr_ind, all_obj_ind = self.make_pair(num_phrase, num_box)

            relation_conn_i = batch_relation_conn[bid]

            """ Language scene graph """
            if len(relation_conn_i) > 0:
                rel_phrase_embed_i = batch_rel_phrase_embed[bid]
                relation_conn_phr_i = torch.Tensor(relation_conn_i)[:, :2].transpose(1, 0).long()

                phrase_embed_i, rel_phrase_embed_i = self.phrase_mps(phrase_embed_i, rel_phrase_embed_i, relation_conn_phr_i, target_id)

                target_embed = phrase_embed_i[target_id]  # ( , 256)
                batch_target_embed.append(torch.unsqueeze(target_embed, 0))

            # objectness_mask = objectness_masks[bid].repeat(num_phrase, 1)

            """ Feature similarity calculation """
            # pred_similarity = self.prediction(box_features[all_obj_ind], phrase_embed_i[all_phr_ind], objectness_mask)
            pred_similarity, reg_offset = self.prediction(box_features[all_obj_ind], phrase_embed_i[all_phr_ind])

            pred_similarity = pred_similarity.reshape(num_phrase, num_box)
            pred_sim = pred_similarity[target_id].unsqueeze(0)
            batch_pred_similarity.append(pred_sim)

            reg_offset_n = reg_offset.reshape(num_phrase, num_box, 6)
            pred_offset = reg_offset_n[target_id].unsqueeze(0)
            batch_reg_offset.append(pred_offset)

            """ Find the topN boxes with the highest scores """
            sorted_score, sorted_ind = torch.sort(pred_similarity, descending=True)
            topN = CONF.MODEL.TOPN
            topN_boxes_ids = sorted_ind[:, :topN]  # num_phr, topN
            topN_boxes_scores = sorted_score[:, :topN]  # num_phr, topN

            topN_target_ids = topN_boxes_ids[target_id]
            batch_topN_target_ids.append(torch.unsqueeze(topN_target_ids, 0))

            if len(relation_conn_i) > 0:
                topN_boxes_ids_numpy = topN_boxes_ids.detach().cpu().numpy()
                topN_boxes_scores_numpy = topN_boxes_scores.detach().cpu().numpy()

                """ Generate union boxes """
                conn_map, sub_obj_union, box_feature_union = generate_union_region_boxes(
                         relation_conn_i, pred_boxes_i, topN_boxes_ids_numpy, topN_boxes_scores_numpy, box_features)

                aggregated_xyz_i = aggregated_xyz[bid]
                seed_features_i = seed_features[bid]
                seed_xyz_i = seed_xyz[bid]

                """ Get union box features """
                nms_union_box_features = self.getboxfeature([sub_obj_union.bbox], box_features, aggregated_xyz_i,
                                                        seed_features_i, seed_xyz_i, box_feature_union)

                conn_map = torch.Tensor(conn_map).cuda().long()
                union_selection_by_id = torch.masked_select(conn_map, conn_map.ge(0))
                union_box_features = torch.index_select(nms_union_box_features, 0, union_selection_by_id.long())

                select_topN_boxes = pred_boxes_i[topN_boxes_ids.reshape(-1)]  # topN*num_phr, 6
                select_topN_reg_ind = topN_boxes_ids.cpu().numpy() + pred_boxes_i.shape[0] * np.arange(num_phrase)[:, None]
                select_topN_offset = reg_offset[select_topN_reg_ind.reshape(-1)]
                select_topN_boxes = self.box_coder.decode(select_topN_offset, select_topN_boxes)  # topN*num_phr, 6

                select_topN_boxes = BoxList(select_topN_boxes, mode="cd")
                select_topN_b = torch.zeros((1, MAX_NUM_P * topN, 6)).cuda()
                select_topN_b[0, :num_phrase * topN, :] = select_topN_boxes.bbox[:, :]
                batch_topN_boxes.append(select_topN_b[:, target_id*topN:(target_id+1)*topN, :])

                features_topN = box_features[topN_boxes_ids.reshape(-1)]
                phr_ind_topN, obj_ind_topN = self.make_pair_topN(num_phrase, topN)

                """ Language scene graph """
                union_box_features, features_topN = self.visual_graph(features_topN, union_box_features, conn_map,
                                                                      topN_boxes_scores, target_id)

                """ Feature similarity calculation """
                pred_similarity_topN, reg_offset_topN = self.prediction_topN(features_topN[obj_ind_topN],
                                                                             phrase_embed_i[phr_ind_topN])
                # reg_offset_t = torch.zeros((1, MAX_NUM_P * topN, 6)).half().cuda()
                # obj_mask = objectness_masks[bid][topN_boxes_ids.reshape(-1)]
                # pred_similarity_topN = self.prediction_topN(features_topN[obj_ind_topN], phrase_embed_i[phr_ind_topN],
                #                                             obj_mask)

                pred_similarity_topN = pred_similarity_topN.reshape(num_phrase, topN)
                pred_similarity_t = pred_similarity_topN[target_id].unsqueeze(0)
                batch_pred_similarity_topN.append(pred_similarity_t)

                reg_topN_offset = reg_offset_topN.reshape(num_phrase, topN, 6)
                reg_offset_t = reg_topN_offset[target_id].unsqueeze(0)
                batch_reg_offset_topN.append(reg_offset_t)

                """ fuse upper 2 results """
                # apply the first stage score into this stage
                pred_sim_det = torch.zeros(1, 256).fill_(-float("inf"))
                pred_similarity_topN = pred_similarity_topN.detach().cpu().numpy()
                pred_similarity_det = pred_similarity_topN * topN_boxes_scores.detach().cpu().numpy()  # num_phr, topN
                pred_similarity_d = torch.tensor(pred_similarity_det)
                for i in range(pred_similarity_d.shape[1]):
                    pred_sim_det[0, topN_boxes_ids[target_id, i]] = pred_similarity_d[target_id, i]

                select_ind_det = pred_similarity_det.argmax(1)[target_id]
                select_ind_det = select_ind_det + topN * target_id
                select_box_det = select_topN_boxes.bbox[select_ind_det]
                select_box_det = torch.unsqueeze(select_box_det, 0)
                select_offset_det = reg_offset_topN[select_ind_det]
                select_offset_det = torch.unsqueeze(select_offset_det, 0)
                pred_box_det = self.box_coder.decode(select_offset_det, select_box_det)

                batch_pred_scores.append(pred_sim_det)
                batch_final_box_det.append(pred_box_det)

        data_dict["batch_pred_similarity"] = torch.cat(tuple(batch_pred_similarity), 0)  # (B, num_box)
        data_dict["batch_reg_offset"] = torch.cat(tuple(batch_reg_offset), 0)  # (B, num_box, 6)
        data_dict["batch_pred_similarity_topN"] = torch.cat(tuple(batch_pred_similarity_topN), 0)
        data_dict["batch_reg_offset_topN"] = torch.cat(tuple(batch_reg_offset_topN), 0)  # (B, topN, 6)
        data_dict["batch_final_box_det"] = torch.cat(tuple(batch_final_box_det), 0)  # (B, 6)
        data_dict["batch_topN_target_ids"] = torch.cat(tuple(batch_topN_target_ids), 0)
        data_dict["batch_target_embed"] = torch.cat(tuple(batch_target_embed), 0)
        data_dict["batch_topN_boxes"] = torch.cat(tuple(batch_topN_boxes), 0)
        data_dict["batch_pred_scores"] = torch.cat(tuple(batch_pred_scores), 0).cuda()  # (B, num_box)

        return data_dict

    def prediction(self, features, phrase_embed):
        fusion_embed = torch.cat((phrase_embed, features), 1)  # num_p*num_b, 256
        cosine_feature = fusion_embed[:, :128] * fusion_embed[:, 128:256]  # num_p*num_b, 128
        delta_feature = fusion_embed[:, :128] - fusion_embed[:, 128:256]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)  # num_p*num_b, 512

        pred_similarity = self.similarity(fusion_embed)
        reg_offset = self.box_reg(fusion_embed)
        return pred_similarity, reg_offset

    def prediction_topN(self, features, phrase_embed):
        fusion_embed = torch.cat((phrase_embed, features), 1)
        cosine_feature = fusion_embed[:, :128] * fusion_embed[:, 128:256]
        delta_feature = fusion_embed[:, :128] - fusion_embed[:, 128:256]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)

        pred_similarity = self.similarity_topN(fusion_embed)
        reg_offset = self.box_reg_topN(fusion_embed)
        return pred_similarity, reg_offset

    def make_pair(self, phr_num: int, box_num: int):
        """
        example  [0 0 0 0 1 1 1 1 2 2 2 2]
                 [0 1 2 3 0 1 2 3 0 1 2 3]
        """
        ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
        ind_phr = ind_phr.reshape(-1)
        ind_box = ind_box.reshape(-1)
        return ind_phr, ind_box

    def make_pair_topN(self, phr_num, topN):
        """
        in topN setting, to pair the phrases and objects. Every phrase have it own topN objects. But they save in previous setting.
        So we need to minus the ids into 0~25
        """
        ind_phr = np.arange(phr_num).repeat(topN)
        ind_box = np.arange(phr_num * topN)
        return ind_phr, ind_box