# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmengine.structures import BaseDataElement
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from yopo.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .topk_hungarian_assigner import TopkHungarianAssigner

def sample_topk_per_gt(pr_inds, gt_inds, cost_matrix, k):
    """
    pr_inds (tensor): tensor of shape (M,)
    gt_inds (tensor): tensor of shape (M,)
    cost_matrix (tensor): tensor of shape (num_targets, num_queries)
    """
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # find topk matches for each gt
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = cost_matrix[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:,None].repeat(1, k)

    # filter to as many matches that gt has
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    scores = torch.cat([s[:c] for c, s in zip(counts, scores)])
    
    # assign query to gt with highest match score
    score_sorted_inds = scores.argsort(descending=False)
    pr_inds3 = pr_inds3[score_sorted_inds]
    gt_inds3 = gt_inds3[score_sorted_inds]

    return pr_inds3, gt_inds3

def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

@TASK_UTILS.register_module()
class O2MHungarianAssigner(TopkHungarianAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(self,
                 *args,
                 o2m_threshold: float = 0.4,
                 coef_box: float = 0.7,
                 coef_cls: float = 0.3,
                 allow_low_quality_matches: bool = True,
                 **kwargs):
        super(O2MHungarianAssigner, self).__init__(*args, **kwargs)
        thresholds = [o2m_threshold]
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))

        self.o2m_thresholds = thresholds
        self.allow_low_quality_matches = allow_low_quality_matches

        self.labels = [0, 1]
        self.coef_cls = coef_cls
        self.coef_box = coef_box
        self.positive_fraction = 0.25

    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor, 
        batch_size_per_image: int,
        bg_label: int,
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + bg_label
        
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, batch_size_per_image, self.positive_fraction, bg_label
        )
        
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def set_low_quality_matches_(self, match_labels, cost_matrix, k=1):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        highest_quality_foreach_gt_inds = cost_matrix.topk(k=k, dim=1)[1]
        match_labels[highest_quality_foreach_gt_inds.flatten()] = 1

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        pred_scores = pred_instances.scores.detach()
        bg_label = pred_scores.shape[1]

        pred_bboxes = pred_instances.bboxes.detach()
        gt_bboxes = gt_instances.bboxes.detach()
        gt_labels = gt_instances.labels.detach()

        pred_scores = pred_scores.detach()
        pred_bboxes = pred_bboxes.detach()
        temp_overlaps = self.iou_calculator(pred_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores.sigmoid()[:, gt_labels].detach()
        cost_matrix = bbox_scores*self.coef_cls + temp_overlaps*self.coef_box

        num_queries = pred_scores.size(0)

        cost_matrix = cost_matrix.T

        num_gt, num_bboxes = gt_bboxes.size(0), pred_scores.size(0)

        assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
        assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)

        assigned_gt_inds[:] = 0

        if num_gt == 0:
            return AssignResult(
                0, assigned_gt_inds, None, labels=assigned_labels)

        assert torch.all(cost_matrix >= 0)

        # cost_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = cost_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.o2m_thresholds[:-1], self.o2m_thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, cost_matrix)
    
        sampled_idxs, sampled_gt_classes = self._sample_proposals(
            matches, match_labels, gt_labels, batch_size_per_image=num_queries,
            bg_label=bg_label
        )

        pos_pr_inds = sampled_idxs[sampled_gt_classes != bg_label]
        pos_gt_inds = matches[pos_pr_inds]
        # [pred_idx, gt_idx]
        # assigned_gt_inds: num_queries
        # label: zero for background, 1+ for gt index
        pos_pr_inds, pos_gt_inds = sample_topk_per_gt(pos_pr_inds, pos_gt_inds, cost_matrix, self.topk)
        assigned_gt_inds[pos_pr_inds] = pos_gt_inds + 1
        assigned_labels[pos_pr_inds] = gt_labels[pos_gt_inds]

        assign_result = AssignResult(
            num_gt, assigned_gt_inds, None, labels=assigned_labels)
        return assign_result
