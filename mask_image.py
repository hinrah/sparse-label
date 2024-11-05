import numpy as np

from constants import Labels


def join_mask_images(images):
    affine = images[0].affine
    shape = images[0].shape
    for image in images[1:]:
        if np.any(affine != image.affine) or image.shape != shape:
            raise RuntimeError("Images you want to join do not have the same shape or affine matrix")

    image_data = [image.mask for image in images]
    stacked_image = np.stack(image_data)

    same = np.all(stacked_image == stacked_image[0], axis=0)

    min_label = np.min(stacked_image, axis=0)
    is_valid = np.all(np.logical_or(stacked_image == min_label, stacked_image == Labels.UNKNOWN), axis=0)

    joined_image = np.ones(shape) * Labels.UNPROCESSED
    joined_image = np.where(is_valid, min_label, joined_image)
    is_invalid = is_valid==False
    if np.sum(is_invalid) > 0:
        print("voxels with different values that are not unknown are set to unknown")
        joined_image = np.where(is_invalid, Labels.UNKNOWN, joined_image)

    new_image = SparseMaskImage(shape, affine)
    new_image.set_mask(joined_image)
    return new_image


class SparseMaskImage:
    def __init__(self, shape, affine):
        self._shape = shape
        self._affine = affine
        self._voxel_center_points = None
        self._mask = np.ones(shape).reshape((-1, 1)) * Labels.UNPROCESSED

    @property
    def mask(self):
        if np.any(self._mask == Labels.UNPROCESSED):
            raise RuntimeError("All voxels need to be set. This error can be prevented by setting ever")
        return self._mask.reshape(self._shape)

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
        voxels_h = homogenous(voxels)
        voxels_w = de_homgenize(voxels_h @ self._affine.T)
        return voxels_w

    @property
    def voxel_center_points(self):
        if self._voxel_center_points is None:
            self._voxel_center_points = self._compute_voxel_center_points()
        return self._voxel_center_points

    def voxel_by_label(self, label):
        return np.nonzero(self._mask == label)[0]

    def label_points_with_label(self, strategy, label):
        points_to_label = self.voxel_center_points[self.voxel_by_label(label), :]
        self._mask[self.voxel_by_label(label)] = strategy.apply(points_to_label)

    def set_mask(self, mask):
        if mask.shape != self._shape:
            raise RuntimeError("The mask does not fit the mask shape")
        self._mask = mask.reshape((-1, 1))


def homogenous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def de_homgenize(points):
    return points[:, :3]
