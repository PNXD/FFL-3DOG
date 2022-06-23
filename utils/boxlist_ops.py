# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from utils.bounding_box import BoxList
from utils.nms import nms_3d_faster as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", require_keep_idx=False):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("cd")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)

    if keep[-1] > boxes.shape[0]:
        kep = sorted(keep, reverse=True)
        for i in kep:
            if i > boxes.shape[0]:
                keep.pop(-1)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    if require_keep_idx:
        return boxlist.convert(mode), keep
    else:
        return boxlist.convert(mode)


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (cx,cy,cz,dx,dy,dz).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,6]. (cx,cy,cz,dx,dy,dz)
      box2: (BoxList) bounding boxes, sized [M,6].(cx,cy,cz,dx,dy,dz)

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    if boxlist1.bbox.device.type != 'cuda':
        boxlist1.bbox = boxlist1.bbox.cuda()

    if boxlist2.bbox.device.type != 'cuda':
        boxlist2.bbox = boxlist2.bbox.cuda()

    box1 = boxlist1.bbox
    box2 = boxlist2.bbox

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    bomin1 = box1[:, :3] - 1 / 2 * box1[:, 3:6]
    bomax1 = box1[:, :3] + 1 / 2 * box1[:, 3:6]
    bomin2 = box2[:, :3] - 1 / 2 * box2[:, 3:6]
    bomax2 = box2[:, :3] + 1 / 2 * box2[:, 3:6]

    rb = torch.min(bomax1[:, None, :3], bomax2[:, :3])
    lt = torch.max(bomin1[:, None, :3], bomin2[:, :3])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou.detach().cpu()


def topN_boxlist_union(boxlist1: BoxList, boxlist2: BoxList, sub_obj_scores_sort=None, sub_obj_scores_sort_id=None,
                       cluster=True):
    """
    give the connection and instance boxes
    calculate the union boxes
    :param boxlist1: sub boxes
    :param boxlist2: obj boxes
    :param sub_obj_scores_sort: sorted union box score
    :param sub_obj_scores_sort_id: sorted union box score index
    :return: nmsed union proposal,
             cluster index for provide the same phrase region that relation can linked to
    """

    box1, box2 = boxlist1, boxlist2

    bo1 = torch.cat(((box1[:, :3] - 1/2 * box1[:, 3:6]), (box1[:, :3] + 1/2 * box1[:, 3:6])), 1)
    bo2 = torch.cat(((box2[:, :3] - 1 / 2 * box2[:, 3:6]), (box2[:, :3] + 1 / 2 * box2[:, 3:6])), 1)

    lt = torch.min(bo1[:, :3], bo2[:, :3])  # [N,M,3]
    rb = torch.max(bo1[:, 3:], bo2[:, 3:])  # [N,M,3]

    ct = 1/2 * (lt + rb)
    dt = (rb - lt)

    union_proposal = torch.cat((ct, dt), dim=1)
    union_proposal = BoxList(union_proposal)
    union_proposal = union_proposal[sub_obj_scores_sort_id]

    # cluster union box
    if cluster:
        union_proposal.add_field('scores', sub_obj_scores_sort)
        union_proposal_nms, keep = boxlist_nms(union_proposal, 0.95, score_field='scores', require_keep_idx=True)
        # retrieval the reduced boxes are clustered by which preserved box
        proposal_cluster = boxlist_iou(union_proposal, union_proposal_nms)
        _, cluster_inx = torch.max(proposal_cluster, dim=1)

        union_proposal = union_proposal_nms
    else:
        cluster_inx = torch.arange(len(union_proposal), dtype=torch.int64, device=box1.device)
        keep = cluster_inx

    return union_proposal, cluster_inx, keep
