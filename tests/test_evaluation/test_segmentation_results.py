import json
import os
import unittest

from sparselabel.constants import ENCODING, DatasetInfo
from sparselabel.evaluation.segmentation_results import SegmentationResults


class TestDatasetConfig(unittest.TestCase):

    def test_evaluate_label_3D(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data_results", "Dataset001_test", "expected_segmentation_results",
                               "segmentation_results.json"), encoding=ENCODING) as file:
            segmentation_result = SegmentationResults()
            raw_segmentation_result = json.load(file)
            segmentation_result.from_json(raw_segmentation_result)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data_results", "Dataset001_test", "expected_segmentation_results",
                               "summarized_segmentation_results.json"), encoding=ENCODING) as file:
            expected_summarized_results = json.load(file)

        actual_summarized_results = {
            "dice_background": segmentation_result.dice_coefficients(DatasetInfo.BACKGROUND),
            "dice_wall": segmentation_result.dice_coefficients(DatasetInfo.WALL),
            "dice_lumen": segmentation_result.dice_coefficients(DatasetInfo.LUMEN),
            "dice_background_median": segmentation_result.dice_coefficients(DatasetInfo.BACKGROUND),
            "dice_wall_median": segmentation_result.dice_coefficients(DatasetInfo.WALL),
            "dice_lumen_median": segmentation_result.dice_coefficients(DatasetInfo.LUMEN),
            "hausdorff_distance_background": segmentation_result.hausdorff_distances(DatasetInfo.BACKGROUND),
            "hausdorff_distance_wall": segmentation_result.hausdorff_distances(DatasetInfo.WALL),
            "hausdorff_distance_lumen": segmentation_result.hausdorff_distances(DatasetInfo.LUMEN),
            "hausdorff_distance_background_median": segmentation_result.hausdorff_distances(DatasetInfo.BACKGROUND),
            "hausdorff_distance_wall_median": segmentation_result.hausdorff_distances(DatasetInfo.WALL),
            "hausdorff_distance_lumen_median": segmentation_result.hausdorff_distances(DatasetInfo.LUMEN),
            "hausdorff_distance_95_background": segmentation_result.hausdorff_distances_95(DatasetInfo.BACKGROUND),
            "hausdorff_distance_95_wall": segmentation_result.hausdorff_distances_95(DatasetInfo.WALL),
            "hausdorff_distance_95_lumen": segmentation_result.hausdorff_distances_95(DatasetInfo.LUMEN),
            "hausdorff_distance_95_background_median": segmentation_result.hausdorff_distances_95(DatasetInfo.BACKGROUND),
            "hausdorff_distance_95_wall_median": segmentation_result.hausdorff_distances_95(DatasetInfo.WALL),
            "hausdorff_distance_95_lumen_median": segmentation_result.hausdorff_distances_95(DatasetInfo.LUMEN),
            "average_countour_distance_background": segmentation_result.average_contour_distances(DatasetInfo.BACKGROUND),
            "average_countour_distance_wall": segmentation_result.average_contour_distances(DatasetInfo.WALL),
            "average_countour_distance_lumen": segmentation_result.average_contour_distances(DatasetInfo.LUMEN),
            "average_countour_distance_background_median": segmentation_result.average_contour_distances(DatasetInfo.BACKGROUND),
            "average_countour_distance_wall_median": segmentation_result.average_contour_distances(DatasetInfo.WALL),
            "average_countour_distance_lumen_median": segmentation_result.average_contour_distances(DatasetInfo.LUMEN),
            "mean_centerline_sensitivity": segmentation_result.centerline_sensitivities(),
            "median_centerline_sensitivity": segmentation_result.centerline_sensitivities(),
            "mean_lumen_background_percentage": segmentation_result.lumen_background_percentages(),
            "median_lumen_background_percentage": segmentation_result.lumen_background_percentages(),
            "num_missed_slices": segmentation_result.num_invalid_results()
        }

        self.assertEqual(json.dumps(expected_summarized_results), json.dumps(actual_summarized_results))
