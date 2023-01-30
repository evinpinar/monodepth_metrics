import numpy as np
import torch
from skimage import feature
from scipy import ndimage

from point_cloud import *
import kaolin as kal


def evaluateDepth3D(predDepth, depth_gt, mask=[]):
    """ An implementation based on Kaolin library. 
    Input depths and masks are torch tensors. """

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


from pytorch3d.ops.knn import knn_points
def new_metrics_eval_(data, config, device, normalize_pcd=True):
    """ Second implementation based on Pytorch3D library.  
    Input depths and masks are torch tensors. """
    depth_gt, depth_pred, intrinsics = data
    depth_gt, depth_pred, mask = validate_depths(depth_gt, depth_pred, config.min_depth, config.max_depth)
    #depth_gt, depth_pred = depth_gt.unsqueeze(0), depth_pred.unsqueeze(0)

    if normalize_pcd:
        gt_pcd, pred_pcd = get_normalized_pcd(depth_gt, depth_pred, intrinsics)
        gt_pcd, pred_pcd = gt_pcd.unsqueeze(0), pred_pcd.unsqueeze(0)
    else:
        gt_pcd, pred_pcd = get_batch_point_cloud(depth_gt.unsqueeze(0), intrinsics, device), get_batch_point_cloud(depth_pred.unsqueeze(0), intrinsics, device)

    gt_pcd_flat, pred_pcd_flat = torch.flatten(gt_pcd, start_dim=1, end_dim=2).to(device).float(), \
                                 torch.flatten(pred_pcd, start_dim=1, end_dim=2).to(device).float()
    x_lengths, x_normals, y_lengths, y_normals = None, None, None, None
    x, x_lengths, x_normals = _handle_pointcloud_input(gt_pcd_flat, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(pred_pcd_flat, y_lengths, y_normals)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)  # completeness
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)  # accuracy
    x_dist = x_nn.dists
    y_dist = y_nn.dists

    cd, completeness, earth_movers = compute_chamfer_distance(gt_pcd_flat, pred_pcd_flat, x_nn, y_nn, x_lengths, y_lengths, x_normals, y_normals)
    iou, f_score = compute_iou_f_score(x_dist, y_dist)  # unsq

    voxel_iou = np.nan
    l2_3d, neg1, neg2 = 0, 0, 0

    return [cd, earth_movers, completeness, iou, f_score]


def validate_depths(gt_depth, pred_depth, min_depth, max_depth):
    pred_depth[torch.isinf(pred_depth)] = max_depth
    pred_depth[torch.isnan(pred_depth)] = min_depth

    masks = torch.logical_and(torch.logical_and(gt_depth, torch.tensor(min_depth).to(gt_depth.device)),
                              torch.logical_and(gt_depth, torch.tensor(max_depth).to(gt_depth.device)))
    if isinstance(masks, np.ndarray):
        masks = masks.astype(np.float32)
    else:
        masks = masks.float()

    # pred_depth = pred_depth * masks
    # gt_depth = gt_depth * masks

    return gt_depth, pred_depth, masks
