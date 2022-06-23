import numpy as np


def nms_3d_faster(boxes, score, overlap_threshold, old_type=False):
    bomin1 = boxes[:, :3] - 1 / 2 * boxes[:, 3:6]
    bomax1 = boxes[:, :3] + 1 / 2 * boxes[:, 3:6]
    x1 = bomin1[:, 0].cpu().detach()
    y1 = bomin1[:, 1].cpu().detach()
    z1 = bomin1[:, 2].cpu().detach()
    x2 = bomax1[:, 0].cpu().detach()
    y2 = bomax1[:, 1].cpu().detach()
    z2 = bomax1[:, 2].cpu().detach()
    score = score

    area = (x2-x1)*(y2-y1)*(z2-z1)
    I = np.argsort(score.cpu())
    pick = []
    while (I.size(0) != 0):
        last = I.size(0)
        i = I[-1]
        pick.append(i.item())
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)
        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / ((area[i] + area[I[:last-1]]) - inter)
        I = np.delete(I, np.concatenate(([last-1], np.where(o > overlap_threshold)[0])))
    return pick
