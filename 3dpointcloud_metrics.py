import numpy as np
import torch
from skimage import feature
from scipy import ndimage

from point_cloud import *
import kaolin as kal


def evaluateDepth3D(predDepth, depth_gt, mask=[]):
    """ Input depths and masks are torch tensors. """


    pred_pc = generate_point_cloud(predDepth)
    gt_pc = generate_point_cloud(depth_gt)

    if len(mask):
        pred_pc = pred_pc[mask == 1]
        gt_pc = gt_pc[mask == 1]
    else:
        pred_pc = torch.flatten(pred_pc, end_dim=1)
        gt_pc = torch.flatten(gt_pc, end_dim=1)

    cd = kal.metrics.point.chamfer_distance(gt_pc, pred_pc)

    iou = kal.metrics.point.iou(gt_pc, pred_pc)

    f_score = kal.metrics.point.f_score(gt_pc, pred_pc)


    return [cd, iou, f_score]