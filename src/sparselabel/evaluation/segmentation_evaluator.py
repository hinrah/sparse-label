import itertools
import math

import numpy as np
from evalutils.stats import mean_contour_distance, hausdorff_distance, percentile_hausdorff_distance, dice_from_confusion_matrix
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from sparselabel.constants import DatasetInfo
from sparselabel.data_handlers.cross_section import CrossSection
from sparselabel.evaluation.metrics import Metrics


class SegmentationEvaluatorAllContoursOn3DLabel:
    def __init__(self, dataset_config):
        self._classes = dataset_config.classes

    def evaluate(self, case):
        return Metrics(identifier=str(case.case_id),
                       dice_coefficients={class_label: np.nan for class_label in self._classes},
                       hausdorff_distances=self._evaluate_metric(case, self._hausdorff_distance),
                       hausdorff_distances_95=self._evaluate_metric(case, self._hausdorff_distance_95),
                       average_contour_distances=self._evaluate_metric(case, self._average_surface_distance),
                       centerline_sensitivity=case.centerline_sensitivity,
                       lumen_background_percentage = case.lumen_background_percentage,
                       is_correct=True)

    def _hausdorff_distance(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.max(distances)

    def _average_surface_distance(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.mean(distances)

    def _hausdorff_distance_95(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.percentile(distances, 95)

    def _evaluate_metric(self, case, metric):
        metrics = {}
        for class_label in self._classes:
            if class_label == DatasetInfo.BACKGROUND:
                metrics[class_label] = np.inf
            if class_label == DatasetInfo.WALL:
                metrics[class_label] = metric(case.outer_mesh_tree, case.true_outer_wall_points())
            if class_label == DatasetInfo.LUMEN:
                metrics[class_label] = metric(case.lumen_mesh_tree, case.true_lumen_points())
        return metrics


class SegmentationEvaluator2DContourOn3DLabel:
    def __init__(self, dataset_config):
        self._classes = dataset_config.classes

    def evaluate(self, truth: CrossSection, case):
        return Metrics(identifier=str(case.case_id) + "__" + str(truth.identifier),
                       dice_coefficients={class_label: np.nan for class_label in self._classes},
                       hausdorff_distances=self._evaluate_metric(truth, case, self._hausdorff_distance),
                       hausdorff_distances_95=self._evaluate_metric(truth, case, self._hausdorff_distance_95),
                       average_contour_distances=self._evaluate_metric(truth, case, self._average_surface_distance),
                       centerline_sensitivity=case.centerline_sensitivity,
                       lumen_background_percentage = case.lumen_background_percentage,
                       is_correct=True)

    def _hausdorff_distance(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.max(distances)

    def _average_surface_distance(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.mean(distances)

    def _hausdorff_distance_95(self, predicted_points, contour_points):
        distances, _ = predicted_points.query(contour_points, k=1)
        return np.percentile(distances, 95)

    def _evaluate_metric(self, truth, case, metric):
        metrics = {}
        for class_label in self._classes:
            if class_label == DatasetInfo.BACKGROUND:
                metrics[class_label] = np.inf
            if class_label == DatasetInfo.WALL:
                metrics[class_label] = metric(case.outer_mesh_tree, truth.outer_wall_points)
            if class_label == DatasetInfo.LUMEN:
                metrics[class_label] = metric(case.lumen_mesh_tree, truth.lumen_points)
        return metrics


class SegmentationEvaluator2DContourOn2DCrossSections:
    def __init__(self, dataset_config, mpr_resolution, mpr_shape):
        self._dataset_config = dataset_config
        self._classes = dataset_config.classes
        self._mpr_resolution = mpr_resolution
        self._mpr_shape = mpr_shape

    def evaluate(self, truth: CrossSection, case):
        mpr = self._mpr(truth, case)
        truth_mask = truth.create_pixel_mask(self._mpr_resolution, self._mpr_shape)
        relevant_mask = self._cut_mask_between_lumen(mpr)

        dice_coefficients = self.evaluate_dice_coefficients(truth_mask, relevant_mask)
        hausdorff_distances = self.evaluate_hausdorff_distances(truth_mask, relevant_mask)
        hausdorff_distances_95 = self.evaluate_hausdorff_distances_95(truth_mask, relevant_mask)
        mean_contour_distances = self.evaluate_mean_contour_distances(truth_mask, relevant_mask)
        all_values = list(dice_coefficients.values()) + list(hausdorff_distances.values()) + list(hausdorff_distances_95.values()) + list(
            mean_contour_distances.values())
        is_correct = np.inf not in all_values

        return Metrics(str(case.case_id) + "__" + str(truth.identifier),
                       dice_coefficients=dice_coefficients,
                       hausdorff_distances=hausdorff_distances,
                       hausdorff_distances_95=hausdorff_distances_95,
                       average_contour_distances=mean_contour_distances,
                       centerline_sensitivity=np.nan,
                       lumen_background_percentage = np.nan,
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
        for class_label in self._classes:
            class_value = self._dataset_config.class_value_by_label(class_label)
            try:
                hausdorff_distances[class_label] = hausdorff_distance(truth == class_value,
                                                                      prediction == class_value,
                                                                      self._mpr_resolution)
            except ValueError:
                hausdorff_distances[class_label] = math.inf
        return hausdorff_distances

    def evaluate_hausdorff_distances_95(self, truth: np.ndarray, prediction: np.ndarray):
        hausdorff_distances = {}
        for class_label in self._classes:
            class_value = self._dataset_config.class_value_by_label(class_label)
            try:
                hausdorff_distances[class_label] = percentile_hausdorff_distance(truth == class_value,
                                                                                 prediction == class_value,
                                                                                 percentile=0.95,
                                                                                 voxelspacing=self._mpr_resolution)
            except ValueError:
                hausdorff_distances[class_label] = math.inf
            except IndexError:
                hausdorff_distances[class_label] = math.inf
        return hausdorff_distances

    def evaluate_mean_contour_distances(self, truth: np.ndarray, prediction: np.ndarray):
        mean_contour_distances = {}
        for class_label in self._classes:
            class_value = self._dataset_config.class_value_by_label(class_label)
            try:
                mean_contour_distances[class_label] = mean_contour_distance(truth == class_value,
                                                                            prediction == class_value,
                                                                            self._mpr_resolution)
            except ValueError:
                mean_contour_distances[class_label] = math.inf
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

        out = map_coordinates(case.prediction_volume, [grid_voxel_coords[..., i] for i in range(3)], order=0, mode='constant',
                              cval=self._dataset_config.background_value)
        return np.round(out).astype(int)

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

    def _get_confusion_matrix(self, truth_mask, prediction_mask):
        confusion_matrix = np.zeros((len(self._classes), len(self._classes)))
        for class_predicted, class_truth in itertools.product(self._classes, self._classes):
            class_value_predicted = self._dataset_config.class_value_by_label(class_predicted)
            class_value_truth = self._dataset_config.class_value_by_label(class_truth)
            confusion_matrix[self._classes.index(class_truth), self._classes.index(class_predicted)] = np.sum(
                np.all(np.stack((prediction_mask == class_value_predicted, truth_mask == class_value_truth)), axis=0))
        return confusion_matrix
