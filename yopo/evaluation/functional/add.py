import torch

def transform_pts(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError('Unsupported shape for T', T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def compute_add(TXO_pred, TXO_gt, points):
    """
    Compute the ADD metric for a batch of predicted and ground truth
    transformations.

    Args:
        TXO_pred (torch.Tensor): Predicted transformations of shape
            (batch_size, 4, 4).
        TXO_gt (torch.Tensor): Ground truth transformations of shape
            (batch_size, 4, 4).
        points (torch.Tensor): Points to be transformed of shape
            (batch_size, n_points, 3).
    Returns:
        torch.Tensor: Distances between predicted and ground truth
            transformations of shape (batch_size, n_points, 3).
    """
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points - TXO_pred_points
    return dists


def compute_add_symmetric(TXO_pred, TXO_gt, points):
    """
    Compute the symmetric ADD metric for a batch of predicted and ground
    truth transformations.

    Args:
        TXO_pred (torch.Tensor): Predicted transformations of shape
            (batch_size, 4, 4).
        TXO_gt (torch.Tensor): Ground truth transformations of shape
            (batch_size, 4, 4).
        points (torch.Tensor): Points to be transformed of shape
            (batch_size, n_points, 3).
    Returns:
        torch.Tensor: Distances between predicted and ground truth
            transformations of shape (batch_size, n_points, 3).
    """
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)
    dists_norm_squared = (dists ** 2).sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    dists = dists[ids_row, assign, ids_col]
    return dists