import numpy as np


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

    @property
    def missed_slices(self):
        return np.sum(np.sum())

    def dice_coefficients(self, class_value=None):
        if class_value:
            return [result.dice_coefficients[class_value] for result in self.valid_results]
        return [list(result.dice_coefficients.values()) for result in self.valid_results]

    def hausdorff_distances(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances[class_value] for result in self.valid_results]
        return [list(result.hausdorff_distances.values()) for result in self.valid_results]

    def hausdorff_distances_95(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances_95[class_value] for result in self.valid_results]
        return [list(result.hausdorff_distances_95.values()) for result in self.valid_results]

    def average_contour_distances(self, class_value=None):
        if class_value:
            return [result.average_contour_distances[class_value] for result in self.valid_results]
        return np.nan_to_num(np.array([list(result.average_contour_distances.values()) for result in self.valid_results]), nan=np.inf)

    def centerline_sensitivities(self):
        return [result.centerline_sensitivity for result in self.valid_results]
