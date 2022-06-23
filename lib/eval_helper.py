import torch
import torch.nn as nn
import numpy as np
import sys
import os

from utils.boxlist_ops import boxlist_iou
from utils.bounding_box import BoxList
from utils.box_util import box3d_iou_batch
from config.config import CONF
from utils.box_util import get_3d_box_batch
from utils.model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()

sys.path.append(os.path.join(os.getcwd()))  # HACK add the lib folder


def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = boxlist_iou(pred_bbox, gt_bbox)

    return iou


def add_heading(pred_bbox):
    heading_angle = 0
    bbox = np.zeros((pred_bbox.shape[0], 7))
    bbox[:, 0:6] = pred_bbox[:, 0:6]
    bbox[:, 6] = heading_angle * -1
    return bbox


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def get_eval(data_dict, use_lang_classifier=True):
    """ Eval module

    Args:
        data_dict: dict
        use_lang_classifier: flag (False/True)
    Returns:
        data_dict: dict
    """

    gt_ref = torch.argmax(data_dict["ref_box_label"], 1)  # B,MAX_NUM_OBJ
    gt_center = data_dict["center_label"].cpu().numpy()  # B,MAX_NUM_OBJ,3
    gt_heading_class = data_dict["heading_class_label"].cpu().numpy()  # B,MAX_NUM_OBJ
    gt_heading_residual = data_dict["heading_residual_label"].cpu().numpy()  # B,MAX_NUM_OBJ
    gt_size_class = data_dict["size_class_label"].cpu().numpy()  # B,MAX_NUM_OBJ
    gt_size_residual = data_dict["size_residual_label"].cpu().numpy()  # B,MAX_NUM_OBJ,3

    pred_bbox = data_dict["precomp_boxes"]
    batch_size, num_proposals, _ = pred_bbox.shape
    batch_final_box_det = data_dict["batch_final_box_det"]
    batch_topN_boxes = data_dict["batch_topN_boxes"]

    ious = []
    multiple = []
    others = []
    pred_bboxes = []
    gt_bboxes = []
    topN_iou1 = []
    topN_iou2 = []
    labels = np.zeros((batch_size, num_proposals))

    for i in range(batch_size):
        gt_ref_idx = gt_ref[i]
        gt_bbox = torch.tensor(DC.param2obb(gt_center[i, gt_ref_idx, 0:3], gt_heading_class[i, gt_ref_idx],
                               gt_heading_residual[i, gt_ref_idx], gt_size_class[i, gt_ref_idx],
                               gt_size_residual[i, gt_ref_idx]))
        gt_bbox = torch.unsqueeze(gt_bbox[:6], 0)
        gt_bbox = BoxList(gt_bbox, mode="cd")
        iou = eval_ref_one_sample(BoxList(torch.unsqueeze(batch_final_box_det[i], 0), mode="cd"), gt_bbox)
        ious.append(iou)

        # topN box iou
        topN_boxes = batch_topN_boxes[i]
        topN_iou = eval_ref_one_sample(BoxList(topN_boxes, mode="cd"), gt_bbox)  # topN,1
        topN_iou_1 = torch.sum(topN_iou >= 0.25)
        topN_iou_2 = torch.sum(topN_iou >= 0.5)
        topN_iou1.append(topN_iou_1)
        topN_iou2.append(topN_iou_2)

        pred_bbbox = construct_bbox_corners(batch_final_box_det[i][0:3].detach().cpu().numpy(), batch_final_box_det[i][3: 6].detach().cpu().numpy())
        gt_bbox = construct_bbox_corners(gt_bbox.bbox[0, 0:3].detach().cpu().numpy(), gt_bbox.bbox[0, 3:6].detach().cpu().numpy())
        pred_bboxes.append(pred_bbbox)
        gt_bboxes.append(gt_bbox)

        # construct the multiple mask
        multiple.append(data_dict["unique_multiple"][i].item())

        # construct the others mask
        flag = 1 if data_dict["object_cat"][i] == 17 else 0
        others.append(flag)

    # store
    data_dict["ref_iou"] = ious
    data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

    data_dict["topN_iou_rate_0.25"] = np.array(topN_iou1)[np.array(topN_iou1) >= 1].shape[0] / np.array(topN_iou1).shape[0]
    data_dict["topN_iou_rate_0.5"] = np.array(topN_iou2)[np.array(topN_iou2) >= 1].shape[0] / np.array(topN_iou2).shape[0]

    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    gt_center_sub = data_dict["ref_center_label"].cpu().numpy()  # B,MAX_NUM_OBJ,3
    gt_heading_class_sub = data_dict["ref_heading_class_label"].cpu().numpy()  # B,K2
    gt_heading_residual_sub = data_dict["ref_heading_residual_label"].cpu().numpy()  # B,K2
    gt_size_class_sub = data_dict["ref_size_class_label"].cpu().numpy()  # B,K2
    gt_size_residual_sub = data_dict["ref_size_residual_label"].cpu().numpy()  # B,K2,3

    for i in range(batch_size):
        # convert the bbox parameters to bbox corners
        gt_box_batch_sub = torch.tensor(DC.param2obb(gt_center_sub[i, :], gt_heading_class_sub[i],
                                                     gt_heading_residual_sub[i], gt_size_class_sub[i],
                                                     gt_size_residual_sub[i, :]))
        gt_box_batch_sub = torch.unsqueeze(gt_box_batch_sub, 0)
        gt_bbox_batch_sub = get_3d_box_batch(gt_box_batch_sub[:, 3:6], gt_box_batch_sub[:, 6], gt_box_batch_sub[:, 0:3])
        pred_box = add_heading(pred_bbox[i].detach().cpu().numpy())
        pred_box = get_3d_box_batch(pred_box[:, 3:6], pred_box[:, 6], pred_box[:, 0:3])
        iousub =  box3d_iou_batch(pred_box, np.tile(gt_bbox_batch_sub, (num_proposals, 1, 1)))
        labels[i, iousub.argmax()] = 1  # treat the bbox with highest iou score as the gt

    cluster_labels = torch.FloatTensor(labels).cuda()

    objectness_preds_batch = torch.argmax(data_dict["objectness_scores"], 2).long()
    objectness_labels_batch = data_dict["objectness_label"].long()
    pred_masks = (objectness_preds_batch == 1).float()
    label_masks = (objectness_labels_batch == 1).float()
    batch_pred_scores = data_dict["batch_pred_scores"]
    cluster_preds = torch.argmax(batch_pred_scores * pred_masks, 1).long().unsqueeze(1).repeat(1, pred_masks.shape[1])
    preds = torch.zeros(pred_masks.shape).cuda()
    preds = preds.scatter_(1, cluster_preds, 1)
    cluster_preds = preds
    cluster_labels = cluster_labels * label_masks

    # compute classification scores
    corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
    labels = torch.ones(corrects.shape[0]).cuda()
    ref_acc = corrects / (labels + 1e-8)

    # store
    data_dict["ref_acc"] = ref_acc.cpu().numpy().tolist()

    # lang
    if use_lang_classifier:
        data_dict["lang_acc"] = (torch.argmax(data_dict["lang_scores"], 1) == data_dict["object_cat"]).float().mean()
    else:
        data_dict["lang_acc"] = torch.zeros(1)[0].cuda()

    # Some other statistics
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"].
                        float())/(torch.sum(data_dict["objectness_mask"].float())+1e-6)
    data_dict["obj_acc"] = obj_acc

    # detection semantic classificatio
    data_dict["pred_mask"] = pred_masks
    sem_cls_label = torch.gather(data_dict["sem_cls_label"], 1, data_dict["object_assignment"])  # select (B,K) from (B,K2)
    sem_cls_pred = data_dict["sem_cls_scores"].argmax(-1)  # B,K
    sem_match = (sem_cls_label == sem_cls_pred).float()
    data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()

    return data_dict
