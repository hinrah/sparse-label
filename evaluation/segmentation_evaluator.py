import itertools
import math

import numpy as np
import trimesh
from evalutils.stats import mean_contour_distance, hausdorff_distance, percentile_hausdorff_distance, dice_from_confusion_matrix
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from constants import Labels
from cross_section import CrossSection
from evaluation.metrics import Metrics


class SegmentationEvaluator2DContourOn3DLabel:
    def __init__(self, classes):
        self._classes = classes

    def evaluate(self, truth: CrossSection, case):
        return Metrics(dice_coefficients={},
                       hausdorff_distances=self._evaluate_metric(truth, case, self._hausdorff_distance),
                       hausdorff_distances_95=self._evaluate_metric(truth, case, self._hausdorff_distance_95),
                       average_contour_distances=self._evaluate_metric(truth, case, self._average_surface_distance),
                       centerline_sensitivity=case.centerline_sensitivity,
                       is_correct=True)

    def _hausdorff_distance(self, mesh, points):
        return np.max(np.abs(trimesh.proximity.signed_distance(mesh, points)))

    def _average_surface_distance(self, mesh, points):
        return np.mean(np.abs(trimesh.proximity.signed_distance(mesh, points)))

    def _hausdorff_distance_95(self, mesh, points):
        return np.percentile(np.abs(trimesh.proximity.signed_distance(mesh, points)), 95)

    def _evaluate_metric(self, truth, case, metric):
        metrics = {}
        for class_value in self._classes:
            if class_value == 0:
                metrics[class_value] = np.inf
            if class_value == 1:
                metrics[class_value] = metric(case.outer_mesh, truth.outer_wall_points)
            if class_value == 2:
                metrics[class_value] = metric(case.lumen_mesh, truth.lumen_points)
        return metrics


class SegmentationEvaluator2DContourOn2DCrossSections:
    def __init__(self, classes, mpr_resolution, mpr_shape):
        self._classes = classes
        self._mpr_resolution = mpr_resolution
        self._mpr_shape = mpr_shape

    def evaluate(self, truth: CrossSection, case):
        mpr = self._mpr(truth, case)
        truth_mask = truth.create_pixel_mask(self._mpr_resolution, self._mpr_shape)
        relevant_mask = self._extract_relevant_mask(mpr)

        dice_coefficients = self.evaluate_dice_coefficients(truth_mask, relevant_mask)
        hausdorff_distances = self.evaluate_hausdorff_distances(truth_mask, relevant_mask)
        hausdorff_distances_95 = self.evaluate_hausdorff_distances_95(truth_mask, relevant_mask)
        mean_contour_distances = self.evaluate_mean_contour_distances(truth_mask, relevant_mask)
        all_values = list(dice_coefficients.values()) + list(hausdorff_distances.values()) + list(hausdorff_distances_95.values()) + list(
            mean_contour_distances.values())
        is_correct = np.inf not in all_values

        return Metrics(dice_coefficients=dice_coefficients,
                       hausdorff_distances=hausdorff_distances,
                       hausdorff_distances_95=hausdorff_distances_95,
                       average_contour_distances=mean_contour_distances,
                       centerline_sensitivity=None,
                       is_correct=is_correct)

    def evaluate_dice_coefficients(self, truth, prediction):
        sorted_dice_coefficients = {}
        confusion_matrix = self._get_confusion_matrix(truth, prediction)
        dice_coefficients = dice_from_confusion_matrix(confusion_matrix)
        for i, class_value in enumerate(self._classes):
            sorted_dice_coefficients[class_value] = dice_coefficients[i]
        return sorted_dice_coefficients

    def evaluate_hausdorff_distances(self, truth: np.ndarray, prediction: np.ndarray):
        hausdorff_distances = {}
        for class_value in self._classes:
            try:
                hausdorff_distances[class_value] = hausdorff_distance(truth == class_value,
                                                                      prediction == class_value,
                                                                      self._mpr_resolution)
            except ValueError:
                hausdorff_distances[class_value] = math.inf
        return hausdorff_distances

    def evaluate_hausdorff_distances_95(self, truth: np.ndarray, prediction: np.ndarray):
        hausdorff_distances = {}
        for class_value in self._classes:
            try:
                hausdorff_distances[class_value] = percentile_hausdorff_distance(truth == class_value,
                                                                                 prediction == class_value,
                                                                                 percentile=0.95,
                                                                                 voxelspacing=self._mpr_resolution)
            except ValueError:
                hausdorff_distances[class_value] = math.inf
            except IndexError:
                hausdorff_distances[class_value] = math.inf
        return hausdorff_distances

    def evaluate_mean_contour_distances(self, truth: np.ndarray, prediction: np.ndarray):
        mean_contour_distances = {}
        for class_value in self._classes:
            try:
                mean_contour_distances[class_value] = mean_contour_distance(truth == class_value,
                                                                            prediction == class_value,
                                                                            self._mpr_resolution)
            except ValueError:
                mean_contour_distances[class_value] = math.inf
        return mean_contour_distances

    def _mpr(self, cross_section, case):
        x_axis = cross_section.plane_x_axis()
        y_axis = cross_section.plane_y_axis()

        x_vals = (np.arange(self._mpr_shape[0]) - self._mpr_shape[0] / 2) * self._mpr_resolution[0]
        y_vals = (np.arange(self._mpr_shape[1]) - self._mpr_shape[1] / 2) * self._mpr_resolution[0]
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid_world_coords = (cross_section.plane_center +
                             x_grid[..., None] * x_axis +
                             y_grid[..., None] * y_axis)

        grid_voxel_coords = np.linalg.inv(case.prediction.affine) @ np.hstack(
            (grid_world_coords.reshape(-1, 3), np.ones((grid_world_coords.shape[0] * grid_world_coords.shape[1], 1)))).T
        grid_voxel_coords = grid_voxel_coords[:3].T.reshape(self._mpr_shape[1], self._mpr_shape[0], 3)

        out = map_coordinates(case.prediction.get_fdata(), [grid_voxel_coords[..., i] for i in range(3)], order=1, mode='constant', cval=0)
        return np.round(out).astype(int)

    def _extract_relevant_mask(self, mask):
        labeled_mask, num_features = ndimage.label(mask != Labels.BACKGROUND)
        center_label = labeled_mask[mask.shape[0] // 2, mask.shape[1] // 2]
        if center_label == 0:
            return np.zeros_like(mask)

        mask = np.where(labeled_mask == center_label, mask, Labels.BACKGROUND)

        labeled_mask, num_features = ndimage.label(mask == Labels.LUMEN)

        if num_features == 0:
            return np.zeros_like(mask)

        distances = []
        center = np.array([mask.shape[0] // 2, mask.shape[1] // 2])
        for i in range(1, num_features + 1):
            points = cKDTree(np.argwhere(labeled_mask == i))
            distance = points.query(center)[0]
            distances.append(distance)
        center_label = np.argmin(distances) + 1

        if center_label == 0:
            raise RuntimeError("this should never happen. If it does, there is a bug.")

        relevant_mask = labeled_mask == center_label
        irrelevant_masks = np.logical_and(labeled_mask != center_label, labeled_mask != 0)

        relevant_pixel = cKDTree(np.argwhere(relevant_mask))
        irrelevant_pixel = cKDTree(np.argwhere(irrelevant_masks))

        all_pixel_coords = np.array(list(np.ndindex(mask.shape)))
        distance_to_correct_lumen = relevant_pixel.query(all_pixel_coords)[0]
        distance_to_incorrect_lumen = irrelevant_pixel.query(all_pixel_coords)[0]

        is_relevant = (distance_to_correct_lumen < distance_to_incorrect_lumen).reshape(mask.shape)

        return np.where(is_relevant, mask, Labels.BACKGROUND)

    def _get_confusion_matrix(self, truth_mask, prediction_mask):
        confusion_matrix = np.zeros((len(self._classes), len(self._classes)))
        for class_predicted, class_truth in itertools.product(self._classes, self._classes):
            confusion_matrix[class_truth, class_predicted] = np.sum(
                np.all(np.stack((prediction_mask == class_predicted, truth_mask == class_truth)), axis=0))
        return confusion_matrix
