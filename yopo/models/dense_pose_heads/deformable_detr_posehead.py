# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from yopo.registry import MODELS
from yopo.structures import SampleList
from yopo.utils import InstanceList, OptInstanceList
from ..layers import inverse_sigmoid
from .detr_posehead import DETRPoseHead


@MODELS.register_module()
class DeformableDETRPoseHead(DETRPoseHead):
    r"""Head of DeformDETR: Deformable DETR: Deformable Transformers for
    End-to-End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 shared_reg_feature: bool = False,
                 use_naive_offset=False,
                 **kwargs) -> None:
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.shared_reg_feature = shared_reg_feature
        self.use_naive_offset = use_naive_offset

        super().__init__(*args, **kwargs)

    def replicate(self, layer, num_layers):
        """Replicate a layer using shared instances or deep copies based on self.share_pred_layer."""
        if self.share_pred_layer:
            return nn.ModuleList([layer] * num_layers)
        else:
            return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        # maintain the same interface as Deformable DETR
        # to utilize the pre-trained model
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.cls_branches = self.replicate(fc_cls, self.num_pred_layer)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        self.reg_branches = self.replicate(reg_branch, self.num_pred_layer)

        reg_feature_branch = []
        if self.shared_reg_feature:
            reg_feature_branch = []
            for _ in range(self.num_reg_fcs):
                reg_feature_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_feature_branch.append(nn.ReLU())
            reg_feature_branch = nn.Sequential(*reg_feature_branch)

            reg_bbox_branch = Linear(self.embed_dims, 4)
            reg_R_branch = Linear(self.embed_dims, 6)
            reg_centers_2d_branch = Linear(self.embed_dims, 2) # embed_dims + bbox
            # reg_centers_2d_branch = Linear(self.embed_dims + 4, 2) # embed_dims + bbox
            reg_z_branch = Linear(self.embed_dims, 1)

            self.reg_feature_branch = self.replicate(reg_feature_branch, self.num_pred_layer)
            self.reg_bbox_branch = self.replicate(reg_bbox_branch, self.num_pred_layer)
            self.reg_R_branch = self.replicate(reg_R_branch, self.num_pred_layer)
            self.reg_centers_2d_branch = self.replicate(reg_centers_2d_branch, self.num_pred_layer)
            self.reg_z_branch = self.replicate(reg_z_branch, self.num_pred_layer)

            # self.reg_branches = nn.ModuleList(
            #     [reg_branch for _ in range(self.num_pred_layer)])

            # self.reg_branches = nn.ModuleList([
            #     copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            # ])

        else:
            all_branches = []
            for i in range(3): # R, center_2d, z
                reg_branch = []
                for _ in range(self.num_reg_fcs):
                    # set the input dim of the first layer
                    embed_dims = self.embed_dims if i != 1 else self.embed_dims + 4
                    if self.use_naive_offset:
                        embed_dims = self.embed_dims

                    reg_branch.append(Linear(embed_dims, embed_dims))
                    reg_branch.append(nn.ReLU())
                if i == 0: # R
                    reg_branch.append(Linear(embed_dims, 6))
                elif i == 1: # center_2d
                    # reg_branch.append(Linear(self.embed_dims, 2))
                    reg_branch.append(Linear(embed_dims, 2))
                elif i == 2: # z
                    reg_branch.append(Linear(embed_dims, 1))
                reg_branch = nn.Sequential(*reg_branch)
                all_branches.append(reg_branch)

            self.reg_R_branch = self.replicate(all_branches[0], self.num_pred_layer)
            self.reg_centers_2d_branch = self.replicate(all_branches[1], self.num_pred_layer)
            self.reg_z_branch = self.replicate(all_branches[2], self.num_pred_layer)


    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        if self.shared_reg_feature:
            for m in self.reg_bbox_branch:
                constant_init(m, 0, bias=0)
            for m in self.reg_centers_2d_branch:
                constant_init(m, 0, bias=0)
        else:
            for m in self.reg_branches:
                constant_init(m[-1], 0, bias=0)
            for m in self.reg_centers_2d_branch:
                constant_init(m[-1], 0, bias=0)
            for m in self.reg_z_branch:
                constant_init(m[-1], 0, bias=6.)

        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        nn.init.constant_(self.reg_centers_2d_branch[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            if self.shared_reg_feature:
                for m in self.reg_bbox_branch:
                    nn.init.constant_(m.bias.data[2:], 0.0)
            else:
                for m in self.reg_branches:
                    nn.init.constant_(m[-1].bias.data[2:], 0.0)
                for m in self.reg_centers_2d_branch:
                    nn.init.constant_(m[-1].bias.data[2:], 0.0)


        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        # if self.as_two_stage:
        #     for m in self.reg_branches:
        #         nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
            - all_layers_outputs_R (Tensor): Direct regression outputs from the
                regression head. Each is a 6D-tensor with [r1, r2] that are first
                two columns of rotation matrix and shape (num_decoder_layers,
                bs, num_queries, 6).
            - all_layers_outputs_centers_2d (Tensor): Sigmoid regression
                outputs from the regression head. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            - all_layers_outputs_z (Tensor): Direct regression outputs from the
                regression head. Each is a 1D-tensor with the directly predicted
                z coordinate and shape (num_decoder_layers, bs, num_queries, 1).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_R = []
        all_layers_outputs_centers_2d = []
        all_layers_outputs_z = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            if self.shared_reg_feature:
                tmp_reg_feature = self.reg_feature_branch[layer_id](hidden_state)
                tmp_reg_bbox_preds = self.reg_bbox_branch[layer_id](tmp_reg_feature)
                tmp_reg_R_preds = self.reg_R_branch[layer_id](tmp_reg_feature)
                # preprocess the R representation
                # rot_c1 = F.normalize(tmp_reg_R_preds[..., :3], dim=-1)
                # rot_c2 = F.normalize(tmp_reg_R_preds[..., 3:], dim=-1)
                # rot_c2 = rot_c2 - torch.sum(
                #     rot_c1 * rot_c2, dim=-1, keepdim=True) * rot_c1
                # tmp_reg_R_preds = torch.cat([rot_c1, rot_c2], dim=-1)
                tmp_reg_centers_2d_preds = self.reg_centers_2d_branch[layer_id](tmp_reg_feature)
                tmp_reg_z_preds = self.reg_z_branch[layer_id](tmp_reg_feature)
            else:
                tmp_reg_bbox_preds = self.reg_branches[layer_id](hidden_state)
                tmp_reg_R_preds = self.reg_R_branch[layer_id](hidden_state)
                # preprocess the R representation
                # rot_c1 = F.normalize(tmp_reg_R_preds[..., :3], dim=-1)
                # rot_c2 = F.normalize(tmp_reg_R_preds[..., 3:], dim=-1)
                # rot_c2 = rot_c2 - torch.sum(
                #     rot_c1 * rot_c2, dim=-1, keepdim=True) * rot_c1
                # tmp_reg_R_preds = torch.cat([rot_c1, rot_c2], dim=-1)


                tmp_reg_z_preds = self.reg_z_branch[layer_id](hidden_state)
            # tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                # tmp_reg_preds += reference
                tmp_reg_bbox_preds += reference

            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                # tmp_reg_preds[..., :2] += reference
                tmp_reg_bbox_preds[..., :2] += reference
            outputs_coord = tmp_reg_bbox_preds.sigmoid()

            if self.use_naive_offset:
                tmp_centers_2d_input = hidden_state
            else:
                tmp_centers_2d_input = torch.cat(
                    [hidden_state, outputs_coord], dim=-1
                )
            tmp_reg_centers_2d_preds = self.reg_centers_2d_branch[layer_id](
                tmp_centers_2d_input)
            tmp_reg_centers_2d_preds = tmp_reg_centers_2d_preds.sigmoid() + \
                outputs_coord[..., :2] - 0.5

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_R.append(tmp_reg_R_preds)
            all_layers_outputs_centers_2d.append(tmp_reg_centers_2d_preds)
            all_layers_outputs_z.append(tmp_reg_z_preds)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_R = torch.stack(all_layers_outputs_R)
        all_layers_outputs_centers_2d = torch.stack(all_layers_outputs_centers_2d)
        all_layers_outputs_z = torch.stack(all_layers_outputs_z)


        return (all_layers_outputs_classes, all_layers_outputs_coords,
                all_layers_outputs_R, all_layers_outputs_centers_2d,
                all_layers_outputs_z)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_R: Tensor, enc_outputs_centers_2d: Tensor,
             enc_outputs_z: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            enc_outputs_R (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (bs, num_feat_points, 6).
            enc_outputs_centers_2d (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (bs, num_feat_points, 2).
            enc_outputs_z (Tensor): Direct regression outputs of each decoder
                layers. Each is a 1D-tensor with the directly predicted z
                coordinate and shape (bs, num_feat_points, 1).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              enc_outputs_R, enc_outputs_centers_2d, enc_outputs_z,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_R_preds: Tensor,
        all_layers_centers_2d_preds: Tensor,
        all_layers_z_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_outputs_R: Tensor,
        enc_outputs_centers_2d: Tensor,
        enc_outputs_z: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            all_layers_R_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_center_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_queries, 1).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            enc_outputs_R (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (bs, num_feat_points, 6).
            enc_outputs_center_2d (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (bs, num_feat_points, 2).
            enc_outputs_z (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (bs, num_feat_points, 1).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(all_layers_cls_scores,
                                         all_layers_bbox_preds,
                                         all_layers_R_preds,
                                         all_layers_centers_2d_preds,
                                         all_layers_z_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            (enc_loss_cls, enc_losses_bbox, enc_losses_iou,
             enc_loss_rotation, enc_loss_pose,
             enc_loss_centers_2d, enc_loss_z) = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    enc_outputs_R, enc_outputs_centers_2d, enc_outputs_z,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            loss_dict['enc_loss_rotation'] = enc_loss_rotation
            loss_dict['enc_loss_pose'] = enc_loss_pose
            loss_dict['enc_loss_centers_2d'] = enc_loss_centers_2d
            loss_dict['enc_loss_z'] = enc_loss_z
        return loss_dict

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_R_preds: Tensor,
                        all_layers_center_2d_preds: Tensor,
                        all_layers_z_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            all_layers_R_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 6D-tensor with
                [r1, r2] that are first two columns of rotation matrix
                and shape (num_decoder_layers, bs, num_queries, 6).
            all_layers_center_2d_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy) and shape
                (num_decoder_layers, bs, num_queries, 2).
            all_layers_z_preds (Tensor): Direct regression
                outputs of each decoder layers. Each is a 1D-tensor with
                the directly predicted z coordinate and shape
                (num_decoder_layers, bs, num_queries, 1).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        R_preds = all_layers_R_preds[-1]
        center_2d_preds = all_layers_center_2d_preds[-1]
        z_preds = all_layers_z_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            R_pred = R_preds[img_id]
            center_2d_pred = center_2d_preds[img_id]
            z_pred = z_preds[img_id]

            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   R_pred, center_2d_pred,
                                                   z_pred, img_meta, rescale)
            result_list.append(results)
        return result_list
