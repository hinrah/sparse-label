import numpy as np

from sparselabel.utils import transform_points

UNPROCESSED = -1


class SparseMaskImage:
    def __init__(self, shape, affine, dataset_config):
        self._dataset_config = dataset_config
        self._shape = shape
        self._affine = affine
        self._voxel_center_points = None
        self._mask = np.ones(shape, dtype=np.int8).reshape((-1, 1)) * UNPROCESSED

    @property
    def mask(self):
        out = self._mask.reshape(self._shape)
        return np.where(out == UNPROCESSED, self._dataset_config.ignore_value, out)

    def set_sparse_mask(self, idx, label):
        label = np.where(label == UNPROCESSED, self._mask[idx], label)  # needed to make ensure unprocessed labels are not overwriting processed labels
        self._mask[idx] = np.where(self._mask[idx] == UNPROCESSED, label, self._mask[idx])
        label = np.where(self._mask[idx] == self._dataset_config.ignore_value, self._dataset_config.ignore_value, label)
        self._mask[idx] = np.where(self._mask[idx] == label, label, self._dataset_config.ignore_value)

    @property
    def affine(self):
        return self._affine

    @property
    def shape(self):
        return self._shape

    def _compute_voxel_center_points(self):
        x = np.arange(self._shape[0])
        y = np.arange(self._shape[1])
        z = np.arange(self._shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        voxels = np.stack((x, y, z), axis=-1).reshape((-1, 3))
        voxels_w = transform_points(voxels, self._affine)
        return voxels_w

    @property
    def voxel_center_points(self):
        if self._voxel_center_points is None:
            self._voxel_center_points = self._compute_voxel_center_points()
        return self._voxel_center_points

    def set_mask(self, mask):
        mask = mask.reshape((-1, 1))

        if mask.shape != self._mask.shape:
            raise RuntimeError("The mask does not fit the mask shape")

        self.set_sparse_mask(range(mask.size), mask)
