import torch
import torch.nn as nn
import numpy as np
import sys
import os

from utils.nn_distance import nn_distance, huber_loss
from lib.softmax_loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box_batch, box3d_iou_batch
from utils.model_util_scannet import ScannetDatasetConfig
from utils.box_coder import reg_encode
from config.config import CONF
from utils.smooth_l1_loss import smooth_l1_loss

sys.path.append(os.path.join(os.getcwd()))  # HACK add the lib folder

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

DC = ScannetDatasetConfig()


def get_loss(data_dict, use_lang_classifier=True, detection=True):
    """ Loss functions

    Args:
        data_dict: dict
        use_lang_classifier: flag (False/True)
        no_detection: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    pos_ratio = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    neg_ratio = torch.sum(objectness_mask.float())/float(total_num_proposal) - pos_ratio
    data_dict["pos_ratio"] = pos_ratio
    data_dict["neg_ratio"] = neg_ratio

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict)
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["center_loss"] = center_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_cls_loss"] = size_cls_loss
        data_dict["size_reg_loss"] = size_reg_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        data_dict["vote_loss"] = torch.zeros(1)[0].cuda()
        data_dict["objectness_loss"] = torch.zeros(1)[0].cuda()
        data_dict["center_loss"] = torch.zeros(1)[0].cuda()
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].cuda()
        data_dict["size_cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["size_reg_loss"] = torch.zeros(1)[0].cuda()
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["box_loss"] = torch.zeros(1)[0].cuda()

    # Reference loss
    ref_loss, cluster_labels, ref_loss_topN, cluster_topN_labels, reg_loss, reg_loss_topN = compute_reference_loss(data_dict)
    data_dict["cluster_labels"] = cluster_labels
    data_dict["ref_loss"] = ref_loss
    data_dict["cluster_topN_labels"] = cluster_topN_labels
    data_dict["ref_loss_topN"] = ref_loss_topN
    data_dict["reg_loss"] = reg_loss
    data_dict["reg_loss_topN"] = reg_loss_topN

    graph_loss = ref_loss + ref_loss_topN + reg_loss + reg_loss_topN
    data_dict["graph_loss"] = graph_loss

    if use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["vote_loss"] + 0.5*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"] \
        + 0.1*data_dict["graph_loss"] + 0.1*data_dict["lang_loss"]
    loss = loss*10  # amplify

    data_dict["loss"] = loss

    return loss, data_dict


def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict["seed_xyz"].shape[0]
    num_seed = data_dict["seed_xyz"].shape[1]  # B,num_seed,3
    vote_xyz = data_dict["vote_xyz"]  # B,num_seed*vote_factor,3
    seed_inds = data_dict["seed_inds"].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict["vote_label_mask"], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict["vote_label"], 1, seed_inds_expand)
    seed_gt_votes = seed_gt_votes+data_dict["seed_xyz"].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    _, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss


def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]
    gt_center = data_dict["center_label"][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    dist1, ind1, _, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict["objectness_scores"]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(data_dict):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = DC.num_heading_bin
    num_size_cluster = DC.num_size_cluster
    mean_size_arr = DC.mean_size_arr

    object_assignment = data_dict["object_assignment"]
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict["center"]
    gt_center = data_dict["center_label"][:, :, 0:3]
    dist1, _, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict["box_label_mask"]
    objectness_label = data_dict["objectness_label"].float()
    centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict["heading_class_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict["heading_scores"].transpose(2,1), heading_class_label)  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict["heading_residual_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict["heading_residuals_normalized"]*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict["size_class_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict["size_scores"].transpose(2,1), size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict["size_residual_label"], 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict["size_residuals_normalized"]*size_label_one_hot_tiled, 2)  # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict["sem_cls_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict["sem_cls_scores"].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def compute_reference_loss(data_dict):
    """ Compute object reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, cluster_labels, ref_loss_topN, cluster_topN_labels
    """

    batch_topN_target_ids = data_dict["batch_topN_target_ids"].cpu()
    # unpack
    cluster_preds = data_dict["batch_pred_similarity"]  # B, num_proposal

    # predicted bbox
    pred_ref = data_dict["batch_pred_similarity"].detach().cpu().numpy()  # B
    pred_center = data_dict["center"].detach().cpu().numpy()  # B,K,3
    pred_heading_class = torch.argmax(data_dict["heading_scores"], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data_dict["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict["ref_center_label"].cpu().numpy()  # B, 3
    gt_heading_class = data_dict["ref_heading_class_label"].cpu().numpy()  # B
    gt_heading_residual = data_dict["ref_heading_residual_label"].cpu().numpy()  # B
    gt_size_class = data_dict["ref_size_class_label"].cpu().numpy()  # B
    gt_size_residual = data_dict["ref_size_residual_label"].cpu().numpy()  # B, 3

    # convert gt bbox parameters to bbox corners
    gt_obb_batch = DC.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                                      gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))
    pos_ind = []
    pred_obb = []
    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                                            pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        labels[i, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
        pos_inds = torch.nonzero(torch.tensor(ious) >= CONF.MODEL.FG_REG_IOU_THRESHOLD)
        pos_ind.append(pos_inds)
        pred_obb.append(pred_obb_batch)

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss
    criterion = SoftmaxRankingLoss()
    ref_loss = criterion(cluster_preds, cluster_labels.float().clone())

    # reg loss
    reg_loss = torch.zeros(1).cuda()
    batch_reg_offset = data_dict["batch_reg_offset"]  # B, num_box, 6
    for i in range(batch_size):
        pred_obb_batch = torch.tensor(pred_obb[i])
        pos_inds = pos_ind[i]
        if len(pos_inds) > 0:
            obj_ind = pos_inds.squeeze(-1)
            regression_pred = batch_reg_offset[i, obj_ind]
            gt_obb_batch = torch.tensor(gt_obb_batch)
            regression_targets = reg_encode(gt_obb_batch[i, :6].unsqueeze(0).repeat(len(obj_ind), 1),
                                            pred_obb_batch[obj_ind, :6])
            reg_loss += CONF.SOLVER.REGLOSS_FACTOR * smooth_l1_loss(
                regression_pred,
                regression_targets.cuda(),
                size_average=True,
                beta=1,
            )

    """topN_loss calculate"""
    cluster_preds_topN = data_dict["batch_pred_similarity_topN"]  # B,topN
    pred_ref_topN = data_dict["batch_pred_similarity_topN"].detach().cpu().numpy()  # B,topN

    # compute the iou score for topN predictd positive ref
    batch_size, topN = cluster_preds_topN.shape
    labels = np.zeros((batch_size, topN))
    pos_ind_topN = []
    pred_obb_topN = []
    for i in range(pred_ref_topN.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = DC.param2obb_batch(pred_center[i, batch_topN_target_ids[i], 0:3], pred_heading_class
                                            [i, batch_topN_target_ids[i]], pred_heading_residual[i, batch_topN_target_ids[i]],
                                            pred_size_class[i, batch_topN_target_ids[i]], pred_size_residual[i, batch_topN_target_ids[i], :])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (topN, 1, 1)))
        labels[i, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
        pos_inds_topN = torch.nonzero(torch.tensor(ious) >= CONF.MODEL.FG_REG_IOU_THRESHOLD)
        pos_ind_topN.append(pos_inds_topN)
        pred_obb_topN.append(pred_obb_batch)

    cluster_topN_labels = torch.FloatTensor(labels).cuda()

    # topN reference loss
    criterion = SoftmaxRankingLoss()
    ref_loss_topN = criterion(cluster_preds_topN, cluster_topN_labels.float().clone())

    # topN reg loss
    reg_loss_topN = torch.zeros(1).cuda()
    batch_reg_offset_topN = data_dict["batch_reg_offset_topN"]  # (B, topN, 6)
    for i in range(batch_size):
        pred_obb_batch = torch.tensor(pred_obb_topN[i])
        pos_inds_topN = pos_ind_topN[i]
        if len(pos_inds_topN) > 0:
            obj_ind = pos_inds_topN.squeeze(-1)
            regression_pred_topN = batch_reg_offset_topN[i, obj_ind]
            gt_obb_batch = torch.tensor(gt_obb_batch)
            regression_targets_topN = reg_encode(gt_obb_batch[i, :6].unsqueeze(0).repeat(len(obj_ind), 1),
                                                 pred_obb_batch[obj_ind, :6])
            reg_loss_topN += CONF.SOLVER.REGLOSS_FACTOR * smooth_l1_loss(
                regression_pred_topN,
                regression_targets_topN.cuda(),
                size_average=True,
                beta=1,
            )

    return ref_loss, cluster_labels, ref_loss_topN, cluster_topN_labels, reg_loss, reg_loss_topN


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss
