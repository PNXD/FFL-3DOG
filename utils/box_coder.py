# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import numpy as np
from config.config import CONF


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_dx = proposals[:, 3] #- proposals[:, 0] + TO_REMOVE
        ex_dy = proposals[:, 4] #- proposals[:, 1] + TO_REMOVE
        ex_dz = proposals[:, 5] #- proposals[:, 0] + TO_REMOVE

        ex_ctr_x = proposals[:, 0] #+ 0.5 * ex_dx
        ex_ctr_y = proposals[:, 1] #+ 0.5 * ex_dy
        ex_ctr_z = proposals[:, 2] #+ 0.5 * ex_dy

        gt_dx = reference_boxes[:, 3]# - reference_boxes[:, 0] + TO_REMOVE
        gt_dy = reference_boxes[:, 4]# - reference_boxes[:, 1] + TO_REMOVE
        gt_dz = reference_boxes[:, 5]  # - proposals[:, 0] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] #+ 0.5 * gt_dx
        gt_ctr_y = reference_boxes[:, 1] #+ 0.5 * gt_dy
        gt_ctr_z = reference_boxes[:, 2] #+ 0.5 * gt_dy

        wcx, wcy, wcz, wdx, wdy, wdz = self.weights
        targets_dcx = wcx * (gt_ctr_x - ex_ctr_x) / ex_dx
        targets_dcy = wcy * (gt_ctr_y - ex_ctr_y) / ex_dy
        targets_dcz = wcz * (gt_ctr_z - ex_ctr_z) / ex_dz

        targets_ddx = wdx * torch.log(gt_dx / ex_dx)
        targets_ddy = wdy * torch.log(gt_dy / ex_dy)
        targets_ddz = wdz * torch.log(gt_dz / ex_dz)

        targets = torch.stack((targets_dcx, targets_dcy, targets_dcz, targets_ddx, targets_ddy, targets_ddz), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        bdx = boxes[:, 3]
        bdy = boxes[:, 4]
        bdz = boxes[:, 5]
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        ctr_z = boxes[:, 2]

        wcx, wcy, wcz, wdx, wdy, wdz = self.weights
        dcx = rel_codes[:, 0::6] / wcx
        dcy = rel_codes[:, 1::6] / wcy
        dcz = rel_codes[:, 2::6] / wcx
        ddx = rel_codes[:, 3::6] / wdx
        ddy = rel_codes[:, 4::6] / wdy
        ddz = rel_codes[:, 5::6] / wdy

        # Prevent sending too large values into torch.exp()
        ddx = torch.clamp(ddx, max=self.bbox_xform_clip)
        ddy = torch.clamp(ddy, max=self.bbox_xform_clip)
        ddz = torch.clamp(ddz, max=self.bbox_xform_clip)

        # apply the regression  adjustment on every coordination
        # the amount of boxes is decided by the regression numbers
        # how many regression how many boxes
        pred_ctr_x = dcx * bdx[:, None] + ctr_x[:, None]
        pred_ctr_y = dcy * bdy[:, None] + ctr_y[:, None]
        pred_ctr_z = dcz * bdz[:, None] + ctr_z[:, None]
        pred_dx = torch.exp(ddx) * bdx[:, None]
        pred_dy = torch.exp(ddy) * bdy[:, None]
        pred_dz = torch.exp(ddz) * bdz[:, None]

        # trans to cx cy cz dx dy dz
        pred_boxes = torch.zeros_like(rel_codes)
        # cx1
        pred_boxes[:, 0::6] = pred_ctr_x
        # cy1
        pred_boxes[:, 1::6] = pred_ctr_y
        # cz1
        pred_boxes[:, 2::6] = pred_ctr_z

        # dx2
        pred_boxes[:, 3::6] = pred_dx
        # dy2
        pred_boxes[:, 4::6] = pred_dy
        # dy2
        pred_boxes[:, 5::6] = pred_dz

        return pred_boxes


def reg_encode(reference_boxes, proposals):
    if proposals.ndim != 2:
        proposals = np.expand_dims(proposals, 0)
    TO_REMOVE = 1  # TODO remove
    ex_dx = proposals[:, 3] + TO_REMOVE
    ex_dy = proposals[:, 4] + TO_REMOVE
    ex_dz = proposals[:, 5] + TO_REMOVE

    ex_ctr_x = proposals[:, 0]
    ex_ctr_y = proposals[:, 1]
    ex_ctr_z = proposals[:, 2]

    gt_dx = reference_boxes[:, 3] + TO_REMOVE
    gt_dy = reference_boxes[:, 4] + TO_REMOVE
    gt_dz = reference_boxes[:, 5] + TO_REMOVE
    gt_ctr_x = reference_boxes[:, 0]
    gt_ctr_y = reference_boxes[:, 1]
    gt_ctr_z = reference_boxes[:, 2]

    wcx, wcy, wcz, wdx, wdy, wdz = CONF.MODEL.BBOX_REG_WEIGHTS
    targets_dcx = wcx * (gt_ctr_x - ex_ctr_x) / ex_dx
    targets_dcy = wcy * (gt_ctr_y - ex_ctr_y) / ex_dy
    targets_dcz = wcz * (gt_ctr_z - ex_ctr_z) / ex_dz

    targets_ddx = wdx * torch.log(gt_dx / ex_dx)
    targets_ddy = wdy * torch.log(gt_dy / ex_dy)
    targets_ddz = wdz * torch.log(gt_dz / ex_dz)

    targets = torch.stack((targets_dcx, targets_dcy, targets_dcz, targets_ddx, targets_ddy, targets_ddz), dim=1)
    return targets
