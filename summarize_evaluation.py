import argparse
import json

import numpy as np
import pandas as pd

from constants import ENCODING, DatasetInfo
from evaluation.segmentation_results import SegmentationResults


def save_results_to_csv(path_to_save, experiements):
    metrics_data = {
        "name": [name for name, result in experiements],
        "dice_wall": [np.mean(result.dice_coefficients(DatasetInfo.WALL)) for name, result in experiements],
        "dice_lumen": [np.mean(result.dice_coefficients(DatasetInfo.LUMEN)) for name, result in experiements],
        "dice_wall_median": [np.median(result.dice_coefficients(DatasetInfo.WALL)) for name, result in experiements],
        "dice_lumen_median": [np.median(result.dice_coefficients(DatasetInfo.LUMEN)) for name, result in experiements],
        "hausdorff_distance_wall": [np.mean(result.hausdorff_distances(DatasetInfo.WALL)) for name, result in experiements],
        "hausdorff_distance_lumen": [np.mean(result.hausdorff_distances(DatasetInfo.LUMEN)) for name, result in experiements],
        "hausdorff_distance_wall_median": [np.median(result.hausdorff_distances(DatasetInfo.WALL)) for name, result in experiements],
        "hausdorff_distance_lumen_median": [np.median(result.hausdorff_distances(DatasetInfo.LUMEN)) for name, result in experiements],
        "hausdorff_distance_95_wall": [np.mean(result.hausdorff_distances_95(DatasetInfo.WALL)) for name, result in experiements],
        "hausdorff_distance_95_lumen": [np.mean(result.hausdorff_distances_95(DatasetInfo.LUMEN)) for name, result in experiements],
        "hausdorff_distance_95_wall_median": [np.median(result.hausdorff_distances_95(DatasetInfo.WALL)) for name, result in experiements],
        "hausdorff_distance_95_lumen_median": [np.median(result.hausdorff_distances_95(DatasetInfo.LUMEN)) for name, result in experiements],
        "average_countour_distance_wall": [np.mean(result.average_contour_distances(DatasetInfo.WALL)) for name, result in experiements],
        "average_countour_distance_lumen": [np.mean(result.average_contour_distances(DatasetInfo.LUMEN)) for name, result in experiements],
        "average_countour_distance_wall_median": [np.median(result.average_contour_distances(DatasetInfo.WALL)) for name, result in experiements],
        "average_countour_distance_lumen_median": [np.median(result.average_contour_distances(DatasetInfo.LUMEN)) for name, result in experiements],
        "mean_centerline_sensitivity": [np.mean(result.centerline_sensitivities()) for name, result in experiements],
        "median_centerline_sensitivity": [np.median(result.centerline_sensitivities()) for name, result in experiements],
        "num_missed_slices": [result.num_invalid_results() for name, result in experiements]
    }
    df = pd.DataFrame(metrics_data)
    df.to_csv(path_to_save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-files', nargs='+', help="file_paths for the files that should be summarized")
    parser.add_argument('-o', help="Path to save the output csv to")
    args = parser.parse_args()

    experiments = []
    for file in args.files:
        with open(file, encoding=ENCODING) as fp:
            segmentation_result = SegmentationResults()
            raw_segmentation_result = json.load(fp)
            segmentation_result.from_json(raw_segmentation_result)
        experiments.append((file, segmentation_result))
    save_results_to_csv(args.o, experiments)


if __name__ == "__main__":
    main()
