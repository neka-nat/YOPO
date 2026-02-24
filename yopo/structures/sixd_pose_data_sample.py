# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData
from .det_data_sample import DetDataSample as BaseDetDataSample


class SixDPoseDataSample(BaseDetDataSample):
    @property
    def gt_instances_T(self) -> InstanceData:
        return self._gt_instances_T

    @gt_instances_T.setter
    def gt_instances_T(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_T', dtype=InstanceData)

    @gt_instances_T.deleter
    def gt_instances_T(self) -> None:
        del self._gt_instances_T

    @property
    def pred_instances_T(self) -> InstanceData:
        return self._pred_instances_T

    @pred_instances_T.setter
    def pred_instances_T(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_T', dtype=InstanceData)

    @pred_instances_T.deleter
    def pred_instances_T(self) -> None:
        del self._pred_instances_T

    @property
    def gt_instances_rotation(self) -> InstanceData:
        return self._gt_instances_rotation

    @gt_instances_rotation.setter
    def gt_instances_rotation(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_rotation', dtype=InstanceData)

    @gt_instances_rotation.deleter
    def gt_instances_rotation(self) -> None:
        del self._gt_instances_rotation

    @property
    def pred_instances_rotation(self) -> InstanceData:
        return self._pred_instances_rotation

    @pred_instances_rotation.setter
    def pred_instances_rotation(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_rotation', dtype=InstanceData)

    @pred_instances_rotation.deleter
    def pred_instances_rotation(self) -> None:
        del self._pred_instances_rotation

    @property
    def gt_instances_center_2d(self) -> InstanceData:
        return self._gt_instances_center_2d

    @gt_instances_center_2d.setter
    def gt_instances_center_2d(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_center_2d', dtype=InstanceData)

    @gt_instances_center_2d.deleter
    def gt_instances_center_2d(self) -> None:
        del self._gt_instances_center_2d

    @property
    def pred_instances_center_2d(self) -> InstanceData:
        return self._pred_instances_center_2d

    @pred_instances_center_2d.setter
    def pred_instances_center_2d(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_center_2d', dtype=InstanceData)

    @pred_instances_center_2d.deleter
    def pred_instances_center_2d(self) -> None:
        del self._pred_instances_center_2d

    @property
    def gt_instances_z(self) -> InstanceData:
        return self._gt_instances_z

    @gt_instances_z.setter
    def gt_instances_z(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_z', dtype=InstanceData)

    @gt_instances_z.deleter
    def gt_instances_z(self) -> None:
        del self._gt_instances_z

    @property
    def pred_instances_z(self) -> InstanceData:
        return self._pred_instances_z

    @pred_instances_z.setter
    def pred_instances_z(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_z', dtype=InstanceData)

    @pred_instances_z.deleter
    def pred_instances_z(self) -> None:
        del self._pred_instances_z