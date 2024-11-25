import argparse
from glob import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate3DSegmentationOnSparse import evaluate_segmentations
from evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn3DLabel, SegmentationEvaluator2DContourOn2DCrossSections
from constants import data_raw


def save_results_to_csv(path_to_save, metrics):
    metrics_data = {
        "name": [metric[0] for metric in metrics],
        "dice_wall": [np.mean(metric[1].dice_coefficients(), axis=0)[1] for metric in metrics],
        "dice_lumen": [np.mean(metric[1].dice_coefficients(), axis=0)[2] for metric in metrics],
        "dice_wall_median": [np.median(metric[1].dice_coefficients(), axis=0)[1] for metric in metrics],
        "dice_lumen_median": [np.median(metric[1].dice_coefficients(), axis=0)[2] for metric in metrics],
        "hausdorff_distance_wall": [np.mean(metric[1].hausdorff_distances(), axis=0)[1] for metric in metrics],
        "hausdorff_distance_lumen": [np.mean(metric[1].hausdorff_distances(), axis=0)[2] for metric in metrics],
        "hausdorff_distance_wall_median": [np.median(metric[1].hausdorff_distances(), axis=0)[1] for metric in metrics],
        "hausdorff_distance_lumen_median": [np.median(metric[1].hausdorff_distances(), axis=0)[2] for metric in metrics],
        "hausdorff_distance_95_wall": [np.mean(metric[1].hausdorff_distances_95(), axis=0)[1] for metric in metrics],
        "hausdorff_distance_95_lumen": [np.mean(metric[1].hausdorff_distances_95(), axis=0)[2] for metric in metrics],
        "hausdorff_distance_95_wall_median": [np.median(metric[1].hausdorff_distances_95(), axis=0)[1] for metric in metrics],
        "hausdorff_distance_95_lumen_median": [np.median(metric[1].hausdorff_distances_95(), axis=0)[2] for metric in metrics],
        "average_countour_distance_wall": [np.mean(metric[1].average_contour_distances(), axis=0)[1] for metric in metrics],
        "average_countour_distance_lumen": [np.mean(metric[1].average_contour_distances(), axis=0)[2] for metric in metrics],
        "average_countour_distance_wall_median": [np.median(metric[1].average_contour_distances(), axis=0)[1] for metric in metrics],
        "average_countour_distance_lumen_median": [np.median(metric[1].average_contour_distances(), axis=0)[2] for metric in metrics],
        "mean_centerline_sensitivity": [np.mean(metric[1].centerline_sensitivities()) for metric in metrics],
        "median_centerline_sensitivity": [np.median(metric[1].centerline_sensitivities()) for metric in metrics],
        "num_missed_slices": [metric[1].num_invalid_results() for metric in metrics]
    }
    df = pd.DataFrame(metrics_data)
    df.to_csv(path_to_save)

evaluators = {
    "3D": SegmentationEvaluator2DContourOn3DLabel(classes=[0, 1, 2]),
    "2D": SegmentationEvaluator2DContourOn2DCrossSections(classes=[0, 1, 2], mpr_resolution=(0.1953125, 0.1953125), mpr_shape=(128,128))
}

def datset_name_for_dataset_id(dataset_id):
    try: 
        dataset_id = int(dataset_id)
    except ValueError:
        return dataset_id
    
    print(os.path.join(data_raw, "Dataset{:03}_*/".format(dataset_id)))

    datasets = list(glob(os.path.join(data_raw, "Dataset{:03}_*/".format(dataset_id))))
    if len(datasets) > 1:
        raise RuntimeError(f"There are more than one dataset with id {dataset_id}")
    
    if len(datasets) == 0:
        raise RuntimeError(f"There are no datasets with id {dataset_id}")

    return Path(datasets[0]).name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    parser.add_argument('-d', nargs='+', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-o', help="[REQUIRED] Output directory for the results.")
    parser.add_argument('-e', help="[REQUIRED] evaluator to use")
    
    args = parser.parse_args()

    metrics = []
    for dataset_id in args.d:
        dataset = datset_name_for_dataset_id(dataset_id) 
        segmentation_results = evaluate_segmentations(dataset, args.n, evaluators[args.e])

        metrics.append((dataset, segmentation_results))

    save_results_to_csv(os.path.join(args.o, f"results_{args.e}.csv"), metrics)
