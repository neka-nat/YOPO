# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yopo.registry import MODELS
from yopo.structures.bbox import bbox_overlaps
from .utils import weighted_loss
from .l2_loss import l2_loss


@weighted_loss
def ard_loss(pred: Tensor,
             target: Tensor,
             eps: float = 1e-6) -> Tensor:
    assert pred.size() == target.size()
    loss = 1 - pred / target
    return loss

@MODELS.register_module()
class ARDLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            # assert weight.shape == pred.shape
            # weight = weight.mean(-1)
            pass
        loss = self.loss_weight * ard_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@weighted_loss
def add_loss(pred: Tensor,
             target: Tensor,
             labels: Optional[Tensor] = None,
             eps: float = 1e-6) -> Tensor:
    """Computes the ADD loss between predicted and target bboxes.
    The ADD loss is defined as the average distance between the predicted
    and target bboxes.
    
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): The learning target of the prediction,
            shape (n, 4).
        labels (Tensor, optional): The labels of the predicted bboxes.
            Defaults to None.
        eps (float): Epsilon to avoid log(0). Defaults to 1e-6.
    Returns:
        Tensor: The ADD loss between the predicted and target bboxes.
    """
    assert pred.size() == target.size()
    pred_R = pred[:, :, :3]
    pred_t = pred[:, :, 3:]
    target_R = target[:, :,  :3]
    target_t = target[:, :, 3:]

    unit_cube = torch.tensor([[-0.5, -0.5, -0.5],
                                [0.5, -0.5, -0.5],
                                [-0.5, 0.5, -0.5],
                                [0.5, 0.5, -0.5],
                                [-0.5, -0.5, 0.5],
                                [0.5, -0.5, 0.5],
                                [-0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5]],
                                dtype=torch.float32).to(pred_R.device)
    unit_cube = unit_cube.unsqueeze(0).repeat(pred_R.size(0), 1, 1)
    
    pred_pts = torch.bmm(unit_cube, pred_R.transpose(1, 2)) + pred_t.transpose(1, 2)
    target_pts = torch.bmm(unit_cube, target_R.transpose(1, 2)) + target_t.transpose(1, 2)

    pred_pts = pred_pts / 1000 # convert from mm to m
    target_pts = target_pts / 1000 # convert from mm to m

    # magic number 177 is the average diameter of the YCB Video dataset
    # add = F.mse_loss(pred_pts, target_pts, reduction='none').mean(dim=(-1, -2)) # / 177.
    add = torch.norm(pred_pts - target_pts, dim=-1, p=2).mean(dim=-1)

    return add

@weighted_loss
def adds_loss(pred: Tensor,
             target: Tensor,
             labels: Optional[Tensor] = None,
             eps: float = 1e-6) -> Tensor:
    """Computes the ADD loss between predicted and target bboxes.
    The ADD loss is defined as the average distance between the predicted
    and target bboxes.
    
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): The learning target of the prediction,
            shape (n, 4).
        labels (Tensor, optional): The labels of the predicted bboxes.
            Defaults to None.
        eps (float): Epsilon to avoid log(0). Defaults to 1e-6.
    Returns:
        Tensor: The ADD loss between the predicted and target bboxes.
    """
    assert pred.size() == target.size()
    pred_R = pred[:, :, :3]
    pred_t = pred[:, :, 3:]
    target_R = target[:, :,  :3]
    target_t = target[:, :, 3:]

    unit_cube = torch.tensor([[-0.5, -0.5, -0.5],
                                [0.5, -0.5, -0.5],
                                [-0.5, 0.5, -0.5],
                                [0.5, 0.5, -0.5],
                                [-0.5, -0.5, 0.5],
                                [0.5, -0.5, 0.5],
                                [-0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5]],
                                dtype=torch.float32).to(pred_R.device)

    unit_cube = unit_cube.unsqueeze(0).repeat(pred_R.size(0), 1, 1)
    
    pred_pts = torch.bmm(unit_cube, pred_R.transpose(1, 2)) + pred_t.transpose(1, 2)
    target_pts = torch.bmm(unit_cube, target_R.transpose(1, 2)) + target_t.transpose(1, 2)

    pred_pts = pred_pts / 1000 # convert from mm to m
    target_pts = target_pts / 1000 # convert from mm to m

    n = pred_pts.size(0)  # Number of predicted objects

    diff = pred_pts[:, None, :, :] - target_pts[:, :, None, :]

    squared_distances = (diff ** 2).sum(dim=-1)  # Shape: [N, M, K]
    # mse = torch.min(((pred_pts[:, None, :, :] - target_pts[:, :, None, :]) ** 2).sum(axis=1), dim=1)[0]
    distances = torch.sqrt(squared_distances)  # Shape: [N, M, K]

    # Step 5: Find the minimum distance for each object
    # If you want the minimum distance between *any* point in pred_pts and *any* point in target_pts
    # for each object, then you need to find the minimum over the two point dimensions.
    # min_distances_per_object = torch.min(distances.view(n, -1), dim=1).values

    # If you want the minimum distance for each point in pred_pts to *any* point in target_pts
    # (i.e., for each pred point, find its closest target point)
    min_dist_pred_to_target = torch.min(distances, dim=2).values # min over target_pts (dim 2)
    # Shape: [num_objects, num_points]

    # If you want the minimum distance for each point in target_pts to *any* point in pred_pts
    # (i.e., for each target point, find its closest pred point)
    # min_dist_target_to_pred = torch.min(distances, dim=1).values # min over pred_pts (dim 1)
    losses = min_dist_pred_to_target.mean(dim=1)  # Average over points for each object

    # losses = mse.mean(axis=-1)

    # dists_squared = (pred_pts.unsqueeze(1) - target_pts.unsqueeze(2)).pow(2)
    # dists = dists_squared
    # dists_norm_squared = dists.sum(dim=-1)
    # assign = dists_norm_squared.argmin(dim=1)
    # ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    # ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    # losses = dists_squared[ids_row, assign, ids_col].mean(dim=(-1, -2))
    return losses

@MODELS.register_module()
class ADDLoss(nn.Module):
    """ADDLoss.

    Computing the Average Distance of the predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 use_unit_cube: bool = True,
                 use_addsym: bool = False,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.use_unit_cube = use_unit_cube
        self.use_addsym = use_addsym
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                centers_2d_pred: Optional[Tensor] = None,
                centers_2d_target: Optional[Tensor] = None,
                z_pred: Optional[Tensor] = None,
                z_target: Optional[Tensor] = None,
                batch_img_metas: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        num_batch = len(batch_img_metas)
        num_queries = pred.shape[0] // num_batch

        R_pred = pred
        R_gt = target
        r1_p, r2_p = torch.split(R_pred, 3, dim=1)
        r1_p = r1_p / torch.norm(r1_p, dim=1, keepdim=True)
        r2_p = r2_p - torch.sum(r1_p * r2_p, dim=1, keepdim=True) * r1_p
        r2_p = r2_p / torch.norm(r2_p, dim=1, keepdim=True)
        r3_p = torch.cross(r1_p, r2_p, dim=1)
        R_pred = torch.stack([r1_p, r2_p, r3_p], dim=1)


        r1_g, r2_g = torch.split(R_gt, 3, dim=1)
        r1_g = r1_g / (torch.norm(r1_g, dim=1, keepdim=True) + self.eps)
        r2_g = r2_g - torch.sum(r1_g * r2_g, dim=1, keepdim=True) * r1_g
        r2_g = r2_g / (torch.norm(r2_g, dim=1, keepdim=True) + self.eps)
        r3_g = torch.cross(r1_g, r2_g, dim=1)
        R_target = torch.stack([r1_g, r2_g, r3_g], dim=1)

        intrinsic = [img_meta['intrinsic'] for img_meta in batch_img_metas]
        if intrinsic is None:
            raise ValueError('intrinsic should not be None')
        if not isinstance(intrinsic, torch.Tensor):
            intrinsic = torch.tensor(intrinsic).to(pred.device)
        intrinsic = intrinsic.view(num_batch, 3, 3)

        K_inv = torch.linalg.inv(intrinsic)
        K_inv = K_inv.repeat_interleave(num_queries, dim=0)

        img_shape = [img_meta['img_shape'] for img_meta in batch_img_metas]
        img_shape = torch.tensor(img_shape).to(pred.device)
        img_shape = img_shape.repeat_interleave(num_queries, dim=0)

        centers_pred = centers_2d_pred * img_shape
        centers_target = centers_2d_target * img_shape

        centers_pred_h = torch.cat([centers_pred,
                                   torch.ones_like(centers_pred[:, :1])],
                                   dim=1)

        depth_pred = torch.exp(z_pred)
        # depth_pred = z_pred * 1800.
        centers_pred_h = centers_pred_h.unsqueeze(-1)  # Shape: [3600, 3, 1]
        # Perform batch matrix multiplication
        t_pred = depth_pred * torch.bmm(K_inv, centers_pred_h).squeeze(-1) # Shape: [3600, 3, 1]

        centers_target_h = torch.cat([centers_target,
                                   torch.ones_like(centers_target[:, :1])],
                                   dim=1)
        depth_target = torch.exp(z_target)
        # depth_target = z_target * 1800.
        # Reshape centers_pred_h to [3600, 3, 1] for batch matrix multiplication
        centers_target_h = centers_target_h.unsqueeze(-1)  # Shape: [3600, 3, 1]
        # Perform batch matrix multiplication
        t_target = depth_target * torch.bmm(K_inv, centers_target_h).squeeze(-1) # Shape: [3600, 3, 1]
        
        pred = torch.cat([R_pred, t_pred.unsqueeze(-1)], dim=2)
        target = torch.cat([R_target, t_target.unsqueeze(-1)], dim=2)

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        # the output shape of add_loss is (N * num_queries)
        weight = weight.mean(-1)
        target = target * weight[:, None, None]

        labels = kwargs.pop('labels', None)

        loss_func = adds_loss if self.use_addsym else add_loss

        loss = self.loss_weight * loss_func(
            pred,
            target,
            labels=labels,
            weight=weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@MODELS.register_module()
class DepthBalanceLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            # assert weight.shape == pred.shape
            # weight = weight.mean(-1)
            pass
        loss = self.loss_weight * l2_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            )
            # **kwargs)
        return loss


@weighted_loss
def pose_loss(pred: Tensor,
             target: Tensor,
             eps: float = 1e-6) -> Tensor:
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)**2
    return loss

@MODELS.register_module()
class OldPoseLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            # assert weight.shape == pred.shape
            # weight = weight.mean(-1)
            pass
        loss = self.loss_weight * pose_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@weighted_loss
def l2pose_loss(pred: Tensor,
             target: Tensor,
             eps: float = 1e-6) -> Tensor:
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)**2
    return loss

@MODELS.register_module()
class L2PoseLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            # assert weight.shape == pred.shape
            # weight = weight.mean(-1)
            pass
        loss = self.loss_weight * l2pose_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@weighted_loss
def l1pose_loss(pred: Tensor,
             target: Tensor,
             eps: float = 1e-6) -> Tensor:
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@MODELS.register_module()
class L1PoseLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        loss = self.loss_weight * l1pose_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@weighted_loss
def chamfer_distance_loss(pred: Tensor,
                         target: Tensor,
                         eps: float = 1e-6) -> Tensor:
    """Computes the Chamfer Distance between two sets of 3D points.
    
    The Chamfer Distance is computed as the average of minimum distances
    from each point in one set to the nearest point in the other set,
    computed bidirectionally.
    
    Args:
        pred (Tensor): Predicted point sets of shape (batch_size, num_points1, 3).
        target (Tensor): Target point sets of shape (batch_size, num_points2, 3).
        eps (float): Epsilon to avoid numerical issues. Defaults to 1e-6.
        
    Returns:
        Tensor: The Chamfer Distance loss for each batch item.
    """
    # pred: (B, N1, 3), target: (B, N2, 3)
    batch_size = pred.size(0)
    
    # Compute pairwise distances
    # pred_expanded: (B, N1, 1, 3), target_expanded: (B, 1, N2, 3)
    pred_expanded = pred.unsqueeze(2)  # (B, N1, 1, 3)
    target_expanded = target.unsqueeze(1)  # (B, 1, N2, 3)
    
    # Compute squared distances: (B, N1, N2)
    squared_distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)
    distances = torch.sqrt(squared_distances + eps)
    
    # Forward direction: For each point in pred, find closest point in target
    min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # (B, N1)
    forward_chamfer = torch.mean(min_dist_pred_to_target, dim=1)  # (B,)
    
    # Backward direction: For each point in target, find closest point in pred
    min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # (B, N2)
    backward_chamfer = torch.mean(min_dist_target_to_pred, dim=1)  # (B,)
    
    # Total Chamfer Distance
    chamfer_dist = forward_chamfer + backward_chamfer
    
    return chamfer_dist


@MODELS.register_module()
class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss.
    
    Computing the Chamfer Distance between two sets of 3D points.
    The Chamfer Distance is computed as the average of minimum distances
    from each point in one set to the nearest point in the other set,
    computed bidirectionally.
    
    Args:
        eps (float): Epsilon to avoid numerical issues. Defaults to 1e-6.
        reduction (str): Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted point sets of shape (batch_size, num_points1, 3).
            target (Tensor): Target point sets of shape (batch_size, num_points2, 3).
            weight (Tensor, optional): The weight of loss for each prediction. 
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
            
        loss = self.loss_weight * chamfer_distance_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@weighted_loss
def chamfer_distance_2d_loss(pred: Tensor,
                            target: Tensor,
                            eps: float = 1e-6) -> Tensor:
    """Computes the 2D Chamfer Distance between two sets of keypoint coordinates.
    
    The 2D Chamfer Distance is computed as the average of minimum distances
    from each keypoint in one set to the nearest keypoint in the other set,
    computed bidirectionally.
    
    Args:
        pred (Tensor): Predicted keypoint sets of shape (batch_size, num_keypoints1, 2).
        target (Tensor): Target keypoint sets of shape (batch_size, num_keypoints2, 2).
        eps (float): Epsilon to avoid numerical issues. Defaults to 1e-6.
        
    Returns:
        Tensor: The 2D Chamfer Distance loss for each batch item.
    """
    # pred: (B, N1, 2), target: (B, N2, 2)
    batch_size = pred.size(0)
    
    # Compute pairwise distances
    # pred_expanded: (B, N1, 1, 2), target_expanded: (B, 1, N2, 2)
    pred_expanded = pred.unsqueeze(2)  # (B, N1, 1, 2)
    target_expanded = target.unsqueeze(1)  # (B, 1, N2, 2)
    
    # Compute squared distances: (B, N1, N2)
    squared_distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)
    distances = torch.sqrt(squared_distances + eps)
    
    # Forward direction: For each keypoint in pred, find closest keypoint in target
    min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # (B, N1)
    forward_chamfer = torch.mean(min_dist_pred_to_target, dim=1)  # (B,)
    
    # Backward direction: For each keypoint in target, find closest keypoint in pred
    min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # (B, N2)
    backward_chamfer = torch.mean(min_dist_target_to_pred, dim=1)  # (B,)
    
    # Total 2D Chamfer Distance
    chamfer_dist = forward_chamfer + backward_chamfer
    
    return chamfer_dist


@MODELS.register_module()
class ChamferDistance2DLoss(nn.Module):
    """2D Chamfer Distance Loss for keypoint coordinates.
    
    Computing the Chamfer Distance between two sets of 2D keypoint coordinates (x,y).
    The Chamfer Distance is computed as the average of minimum distances
    from each keypoint in one set to the nearest keypoint in the other set,
    computed bidirectionally.
    
    Args:
        eps (float): Epsilon to avoid numerical issues. Defaults to 1e-6.
        reduction (str): Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        normalize (bool): Whether to normalize coordinates. Defaults to False.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 normalize: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted keypoint sets of shape (batch_size, num_keypoints1, 2).
            target (Tensor): Target keypoint sets of shape (batch_size, num_keypoints2, 2).
            weight (Tensor, optional): The weight of loss for each prediction. 
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        
        # Optional normalization for keypoint coordinates
        if self.normalize:
            # Normalize to [0, 1] range if coordinates are in pixel space
            pred_norm = pred.clone()
            target_norm = target.clone()
            
            # Assume coordinates are in image space and normalize by image dimensions
            # This can be extended to take image dimensions as input if needed
            if 'img_shape' in kwargs:
                img_h, img_w = kwargs['img_shape']
                pred_norm[..., 0] = pred_norm[..., 0] / img_w
                pred_norm[..., 1] = pred_norm[..., 1] / img_h
                target_norm[..., 0] = target_norm[..., 0] / img_w
                target_norm[..., 1] = target_norm[..., 1] / img_h
            
            pred = pred_norm
            target = target_norm
            
        loss = self.loss_weight * chamfer_distance_2d_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


def _to_rotation_matrix_6d(v6d: Tensor) -> Tensor:
    """Construct a rotation matrix from a 6D representation."""
    r1, r2 = torch.split(v6d, 3, dim=1)
    r1 = F.normalize(r1, p=2, dim=1)

    dot_product = torch.sum(r1 * r2, dim=1, keepdim=True)
    r2_orthogonal = r2 - dot_product * r1
    r2 = F.normalize(r2_orthogonal, p=2, dim=1)

    r3 = torch.cross(r1, r2, dim=1)
    return torch.stack([r1, r2, r3], dim=2)

def _to_rotation_matrix_9d(v9d: Tensor) -> Tensor:
    """Construct a rotation matrix from a 9D representation."""
    m = v9d.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    return torch.matmul(u, vt)

def _geodesic_angle(R1: Tensor, R2: Tensor, eps: float = 1e-6) -> Tensor:
    """Geodesic angle ‖log(R1 R2ᵀ)‖₂ for two rotation matrices."""
    R_err = torch.bmm(R1, R2.transpose(1, 2))
    trace = torch.einsum('bii->b', R_err)
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)

@weighted_loss
def so3_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Computes the geodesic distance loss for 3D rotations.
    The loss is the angle of the error rotation matrix, which is inspired by
    "On the Continuity of Rotation Representations in Neural Networks".

    Args:
        pred (Tensor): Predicted rotations as 6D vectors (first two columns
            of the rotation matrix), shape (n, 6).
        target (Tensor): The learning target of the prediction, shape (n, 6).
        eps (float): Epsilon to avoid numerical issues. Defaults to 1e-7.

    Returns:
        Tensor: The geodesic distance loss.
    """
    assert pred.size(0) == target.size(0), f"Batch size mismatch: {pred.size()[0]} vs {target.size()[0]}"

    R_target = _to_rotation_matrix_6d(target)
    if pred.size(1) == 6:
        R_pred = _to_rotation_matrix_6d(pred)
    elif pred.size(1) == 9:
        R_pred = _to_rotation_matrix_9d(pred)

    angle = _geodesic_angle(R_pred, R_target, eps)
    # l2 distance between the two rotation matrices
    # angle = torch.norm(R_pred - R_target, p='fro', dim=(1, 2)) # / np.sqrt(2)
    # angle = angle.clamp(min=eps)
    return angle

@MODELS.register_module()
class Rotation3DLoss(nn.Module):
    """Rotation3DLoss.

    Computing the rotation loss between a set of predicted bboxes and target bboxes.

    Args:
        distance_type (str): Type of distance to compute. Options are 'l1', 'l2', 'geodesic'.
            Defaults to 'geodesic'.
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 distance_type: str = 'geodesic',
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.distance_type = distance_type
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted rotation matrices of shape (n, M),
                where M could be 9 (for 3x3 matrices), 6 (first two columns of 3x3 matrix),
                or 4 (quaternion).
            target (Tensor): The learning target of the prediction,
                shape (n, M).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0

        if weight.shape[1] in [6, 9]:
            weight = weight.mean(-1)

        loss = self.loss_weight * so3_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
