import numpy as np
import trimesh

from cross_section import CrossSection
from evaluation.metrics import Metrics


class SegmentationEvaluator2DContourOn3DLabel:
    def __init__(self, classes):
        self._classes = classes

    def evaluate(self, truth: CrossSection, case):
        return Metrics(dice_coefficients={},
                       hausdorff_distances=self._evaluate_metric(truth, case, self._hausdorff_distance),
                       hausdorff_distances_95=self._evaluate_metric(truth, case, self._hausdorff_distance_95),
                       average_contour_distances=self._evaluate_metric(truth, case, self._average_surface_distance))

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
