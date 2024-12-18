import numpy as np
from dataclasses import asdict

from sparselabel.evaluation.metrics import Metrics


class SegmentationResults:
    def __init__(self):
        self._results = []

    def add(self, result):
        self._results.append(result)

    @property
    def valid_results(self):
        return [result for result in self._results if result.is_correct]

    def num_invalid_results(self):
        return len(self._results) - len(self.valid_results)

    @property
    def mean_dice_coefficients(self):
        return [np.mean(list(result.dice_coefficients.values())) for result in self.valid_results]

    @property
    def mean_hausdorff_distance(self):
        return [np.mean(list(result.hausdorff_distances.values())) for result in self.valid_results]

    @property
    def mean_average_contour_distances(self):
        return [np.mean(list(result.average_contour_distances.values())) for result in self.valid_results]

    def dice_coefficients(self, class_value=None):
        if class_value:
            return [result.dice_coefficients.get(class_value, 0) for result in self.valid_results]
        return [list(result.dice_coefficients.values()) for result in self.valid_results]

    def hausdorff_distances(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances.get(class_value, np.inf) for result in self.valid_results]
        return [list(result.hausdorff_distances.values()) for result in self.valid_results]

    def hausdorff_distances_95(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances_95.get(class_value, np.inf) for result in self.valid_results]
        return [list(result.hausdorff_distances_95.values()) for result in self.valid_results]

    def average_contour_distances(self, class_value=None):
        if class_value:
            return [result.average_contour_distances.get(class_value, np.inf) for result in self.valid_results]
        return np.nan_to_num(np.array([list(result.average_contour_distances.values()) for result in self.valid_results]), nan=np.inf)

    def centerline_sensitivities(self):
        return [result.centerline_sensitivity for result in self.valid_results]

    def _convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def to_json(self):
        return self._convert_to_serializable([asdict(result) for result in self._results])

    def from_json(self, json_dict):
        self._results = [Metrics(**one_result) for one_result in json_dict]
