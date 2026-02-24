# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from yopo.registry import MODELS
from yopo.structures import OptSampleList
from yopo.utils import OptConfigType
from ...layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)
from ..deformable_detr import DeformableDETR


@MODELS.register_module()
class LiftingDeformablePoseDETR(DeformableDETR):
    r"""Implementation of `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    Args:
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        with_box_refine (bool, optional): Whether to refine the references
            in the decoder. Defaults to `False`.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 4.
    """
    def __init__(self, *args: OptConfigType,
                 num_keypoints: int = 73,
                  **kwargs: OptConfigType) -> None:
        super(LiftingDeformablePoseDETR, self).__init__(*args, **kwargs)
        self.num_keypoints = num_keypoints

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `yopo/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_coord`. They are both `None` when 'as_two_stage'
              is `False`. The dict is empty when `self.training` is `False`.
        """
        batch_size, _, c = memory.shape
    
        output_memory, output_proposals = \
            self.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        if self.bbox_head.shared_reg_feature:
            tmp_reg_feature = self.bbox_head.reg_feature_branch[
                self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_bbox_branch[
                self.decoder.num_layers](tmp_reg_feature)
            enc_outputs_R = self.bbox_head.reg_R_branch(
                tmp_reg_feature)
            enc_outputs_centers_2d = self.bbox_head.reg_centers_2d_branch(
                tmp_reg_feature)
            enc_outputs_z = self.bbox_head.reg_z_branch(
                tmp_reg_feature)
        else:
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory)
            enc_outputs_R = self.bbox_head.reg_R_branch[
                self.decoder.num_layers](output_memory)
            enc_outputs_centers_2d = self.bbox_head.reg_centers_2d_branch[
                self.decoder.num_layers](output_memory)
            enc_outputs_z = self.bbox_head.reg_z_branch[
                self.decoder.num_layers](output_memory)

        enc_outputs_coord_unact = enc_outputs_coord_unact + output_proposals
        enc_outputs_centers_2d = enc_outputs_centers_2d + output_proposals[:, :, :2]

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        enc_outputs_centers_2d = enc_outputs_centers_2d.sigmoid()


        # We only use the first channel in enc_outputs_class as foreground,
        # the other (num_classes - 1) channels are actually not used.
        # Its targets are set to be 0s, which indicates the first
        # class (foreground) because we use [0, num_classes - 1] to
        # indicate class labels, background class is indicated by
        # num_classes (similar convention in RPN).
        # See https://github.com/open-mmlab/mmdetection/blob/master/yopo/models/dense_heads/deformable_detr_head.py#L241 # noqa
        # This follows the official implementation of Deformable DETR.
        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        pos_trans_out = self.pos_trans_fc(
            self.get_proposal_pos_embed(topk_coords_unact))
        pos_trans_out = self.pos_trans_norm(pos_trans_out)
        query_pos, query = torch.split(pos_trans_out, c, dim=2)


        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            enc_outputs_R=enc_outputs_R,
            enc_outputs_centers_2d=enc_outputs_centers_2d,
            enc_outputs_z=enc_outputs_z) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
