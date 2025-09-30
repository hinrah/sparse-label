import math

import numpy as np
from evalutils.stats import hausdorff_distance
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

from sparselabel.constants import DatasetInfo
from sparselabel.data_handlers.cross_section import CrossSection
from sparselabel.evaluation.parameters import Parameters


# pylint: disable=too-many-locals, duplicate-code
class ParameterExtractor:
    def __init__(self, dataset_config, mpr_resolution, mpr_shape):
        self._dataset_config = dataset_config
        self._classes = dataset_config.classes
        self._mpr_resolution = mpr_resolution
        self._mpr_shape = mpr_shape

    def evaluate(self, truth: CrossSection, case):
        mpr = np.round(self._mpr(truth, case.prediction_volume, case.prediction.affine)).astype(int)
        relevant_mask = self._cut_mask_between_lumen(mpr)
        vessel_wall_thickness = self.measure_vessel_wall_thickness(relevant_mask)
        vessel_wall_thickness_manual = self.contour_based_wall_thickness(truth.inner_contour_points, truth._outer_wall_contour_points)

        manual_mask = truth.create_pixel_mask(self._mpr_resolution, self._mpr_shape)
        lumen_diameter = self.lumen_diameter(relevant_mask)
        lumen_diameter_manual = self.lumen_diameter(manual_mask)

        if case.image_data.shape[0] == 4:
            velocity = case.image_data[1:]
            throughplane_component = np.tensordot(truth.plane_normal, velocity, axes=(0, 0))[0]
            throughplane_mpr = self._mpr(truth, throughplane_component, case.prediction.affine)
            max_vel = self.max_vel(relevant_mask, throughplane_mpr)
            max_vel_manual = self.max_vel(manual_mask, throughplane_mpr)

            flow = self.net_flow(relevant_mask, throughplane_mpr)
            flow_manual = self.net_flow(manual_mask, throughplane_mpr)
        else:
            max_vel = max_vel_manual = flow = flow_manual = None

        return Parameters(identifier=str(case.case_id) + "__" + str(truth.identifier),
                          vessel_wall_thickness=vessel_wall_thickness,
                          vessel_wall_thickness_manual=vessel_wall_thickness_manual,
                          lumen_diameter=lumen_diameter,
                          lumen_diameter_manual=lumen_diameter_manual,
                          max_vel=max_vel,
                          max_vel_manual=max_vel_manual,
                          flow=flow,
                          flow_manual=flow_manual,
                          is_correct=vessel_wall_thickness != math.inf)

    def max_vel(self, mask, velocity_magnitude):
        if np.sum(np.astype(mask == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN), int)) == 0:
            return None
        return np.abs(np.astype(mask == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN), int) * velocity_magnitude).max()

    def net_flow(self, mask, throughplane_flow):
        if np.sum(np.astype(mask == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN), int)) == 0:
            return None
        return np.abs(np.sum(np.astype(mask == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN), int) * throughplane_flow) * (
                self._mpr_resolution[0] * self._mpr_resolution[1]))

    def measure_vessel_wall_thickness(self, prediction: np.ndarray):
        try:
            return hausdorff_distance(prediction == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN),
                                      prediction == self._dataset_config.class_value_by_label(DatasetInfo.WALL),
                                      self._mpr_resolution)
        except (ValueError, KeyError):
            return math.inf

    def contour_based_wall_thickness(self, inner, outer):
        if inner is None or outer is None:
            return math.inf
        return max(directed_hausdorff(inner, outer)[0], directed_hausdorff(outer, inner)[0])

    def lumen_diameter(self, mask: np.ndarray):
        area = np.sum(mask == self._dataset_config.class_value_by_label(DatasetInfo.LUMEN))
        area *= self._mpr_resolution[0] * self._mpr_resolution[1]
        return 2 * np.sqrt(area / np.pi)

    def _mpr(self, cross_section, volume, affine):
        x_axis = cross_section.plane_x_axis
        y_axis = cross_section.plane_y_axis

        x_vals = (np.arange(self._mpr_shape[0]) - self._mpr_shape[0] / 2) * self._mpr_resolution[0]
        y_vals = (np.arange(self._mpr_shape[1]) - self._mpr_shape[1] / 2) * self._mpr_resolution[0]
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid_world_coords = (cross_section.plane_center +
                             x_grid[..., None] * x_axis +
                             y_grid[..., None] * y_axis)

        grid_voxel_coords = np.linalg.inv(affine) @ np.hstack(
            (grid_world_coords.reshape(-1, 3), np.ones((grid_world_coords.shape[0] * grid_world_coords.shape[1], 1)))).T
        grid_voxel_coords = grid_voxel_coords[:3].T.reshape(self._mpr_shape[1], self._mpr_shape[0], 3)

        out = map_coordinates(volume, [grid_voxel_coords[..., i] for i in range(3)], order=0, mode='constant',
                              cval=self._dataset_config.background_value)
        return out

    def _extract_center_component(self, mask):
        labeled_mask, _ = ndimage.label(mask != self._dataset_config.background_value)

        center_label = labeled_mask[mask.shape[0] // 2, mask.shape[1] // 2]
        if center_label == 0:
            raise ValueError

        return np.where(labeled_mask == center_label, mask, self._dataset_config.background_value)

    def _classify_pixel_by_lumen(self, mask):
        labeled_mask, num_features = ndimage.label(mask == self._dataset_config.lumen_value)
        if num_features == 0:
            raise ValueError

        distances = []
        center = np.array([mask.shape[0] // 2, mask.shape[1] // 2])
        for i in range(1, num_features + 1):
            points = cKDTree(np.argwhere(labeled_mask == i))
            distance = points.query(center)[0]
            distances.append(distance)
        center_label = np.argmin(distances) + 1

        relevant_mask = labeled_mask == center_label
        irrelevant_masks = np.logical_and(labeled_mask != center_label, labeled_mask != 0)

        relevant_pixel = cKDTree(np.argwhere(relevant_mask))
        irrelevant_pixel = cKDTree(np.argwhere(irrelevant_masks))

        return relevant_pixel, irrelevant_pixel

    def _cut_mask_between_lumen(self, mask):
        try:
            mask = self._extract_center_component(mask)
            relevant_pixel, irrelevant_pixel = self._classify_pixel_by_lumen(mask)
            mask = self._extract_relevant_mask(mask, irrelevant_pixel, relevant_pixel)
        except ValueError:
            return np.ones_like(mask) * self._dataset_config.background_value

        return mask

    def _extract_relevant_mask(self, mask, irrelevant_pixel, relevant_pixel):
        all_pixel_coords = np.array(list(np.ndindex(mask.shape)))
        distance_to_correct_lumen = relevant_pixel.query(all_pixel_coords)[0]
        distance_to_incorrect_lumen = irrelevant_pixel.query(all_pixel_coords)[0]
        is_relevant = (distance_to_correct_lumen < distance_to_incorrect_lumen).reshape(mask.shape)
        relevant_mask = np.where(is_relevant, mask, self._dataset_config.background_value)
        return relevant_mask
