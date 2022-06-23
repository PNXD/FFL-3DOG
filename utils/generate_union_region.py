import numpy as np
import torch
import sys
import os

from utils.boxlist_ops import topN_boxlist_union

sys.path.append(os.path.join(os.getcwd()))


def generate_union_region_boxes(relation_conn, pred_boxes, topN_boxes_ids, topN_boxes_scores, box_features):
    """
    To generate the union bbox
    :param relation_conn: list [[1,2],[1,2]]
    :param pred_boxes: boxlist. boxes
    :param topN_boxes_ids, MxN, M is the number of phrases, N is the number of topN boxes
    :return:
    conn_map: nparray.   (num_phrases * topN, num_phrases * topN). -1 denote no connection.
    0~M, denote index of the union region sorted in phrsbj2obj_union.
    phrsbj2obj_union: the union region lab.
    box_feature_union: initial features of the union box.
    """

    # construct the global connection map
    num_phrases, topN = topN_boxes_ids.shape
    conn_map = np.zeros((num_phrases * topN, num_phrases * topN)) - 1

    # store the relationship between phrases in the connection map
    for rel in relation_conn:
        conn_map[rel[0]*topN:(rel[0]+1)*topN, rel[1]*topN:(rel[1]+1)*topN] = 1

    conn_sub_topN, conn_obj_topN = np.where(conn_map == 1)

    conn_sub_topN_ids = conn_sub_topN // topN  # num_rel*topN
    conn_obj_topN_ids = conn_obj_topN // topN  # num_rel*topN

    # get the ids of the topN boxes for each subject and object
    conn_obj_topN_select = np.tile(np.arange(topN), int(conn_obj_topN.shape[0]/topN))
    sub_topN_bbox_ids = topN_boxes_ids[conn_sub_topN_ids, conn_sub_topN % topN]
    obj_topN_bbox_ids = topN_boxes_ids[conn_obj_topN_ids, conn_obj_topN_select]

    # get the scores for the topN boxes for each subject and object
    sub_topN_scores = topN_boxes_scores[conn_sub_topN_ids, conn_sub_topN % topN]
    obj_topN_scores = topN_boxes_scores[conn_obj_topN_ids, conn_obj_topN_select]
    sub_obj_scores = sub_topN_scores * obj_topN_scores

    # get the parameters of the topN boxes for each subject and object
    pred_boxes_sbj_topN = pred_boxes[sub_topN_bbox_ids.astype(np.int32)]
    pred_boxes_obj_topN = pred_boxes[obj_topN_bbox_ids.astype(np.int32)]

    sub_obj_scores = torch.FloatTensor(sub_obj_scores).cuda()
    sub_obj_scores_sort, sub_obj_scores_sort_id = torch.sort(sub_obj_scores, descending=True)

    sub_obj_union, cluster_idx, keep = topN_boxlist_union(pred_boxes_sbj_topN, pred_boxes_obj_topN,
                                                          sub_obj_scores_sort, sub_obj_scores_sort_id, True)

    box_feature_sbj = box_features[sub_topN_bbox_ids.astype(np.int32)][sub_obj_scores_sort_id][keep]
    box_feature_obj = box_features[obj_topN_bbox_ids.astype(np.int32)][sub_obj_scores_sort_id][keep]
    box_feature_union = torch.unsqueeze(box_feature_sbj, 1) + torch.unsqueeze(box_feature_obj, 1)  # num_nms*1*128

    # reverse the sort. cluster_idx here records the index of each union box after they corresponds to nms.
    # we need to resort the cluster_idx to match it into conn_map
    phrsbj2obj_scores_sort_id_sort = torch.argsort(sub_obj_scores_sort_id)
    keep_inds_mapping = cluster_idx[phrsbj2obj_scores_sort_id_sort]
    conn_map[conn_sub_topN, conn_obj_topN] = keep_inds_mapping.detach().cpu().numpy()

    return conn_map, sub_obj_union, box_feature_union


if __name__ == '__main__':

    pass