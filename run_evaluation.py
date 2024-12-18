import argparse
import json
import os

import numpy as np
import pandas as pd

from dataset_config import DatasetConfig
from evaluation.evaluate3DSegmentationOnSparse import evaluate_segmentations
from evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn3DLabel, SegmentationEvaluator2DContourOn2DCrossSections


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
    "3D": SegmentationEvaluator2DContourOn3DLabel,
    "2D": SegmentationEvaluator2DContourOn2DCrossSections,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    parser.add_argument('-d', nargs='+', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-e', help="[REQUIRED] evaluator to use")
    parser.add_argument('-res_2d', type=float, help="[Required] This is the resolution of the mpr for 2D evaluation in mm.")
    parser.add_argument('-fow_2d', type=float, help="[Required] This is the field of view of the mpr for 2D evaluation in mm.")
    parser.add_argument('-psp', default="nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4",
                        help="[Optional] the sub_path for the prediction. "
                             "The default case is nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4")
    parser.add_argument('-fp', default="Tr",
                        help="[Optionial] folder postfix to distinct different datasets. The default is Tr as in imagesTr, labelsTr. "
                             "This can be changed to evaluate different test datasets.")
    args = parser.parse_args()

    for dataset_id in args.d:
        dataset_config = DatasetConfig(dataset_id, folder_postfix=args.fp, prediction_sub_path=args.psp)
        if args.e == "2D":
            size = int(args.fow_2d // args.res_2d)
            evaluator = SegmentationEvaluator2DContourOn2DCrossSections(dataset_config, (args.res_2d, args.res_2d), (size, size))
        else:
            evaluator = SegmentationEvaluator2DContourOn3DLabel(dataset_config)
        segmentation_results = evaluate_segmentations(dataset_config, args.n, evaluator)
        with open(os.path.join(dataset_config.prediction_path, f"evaluation_results_{args.e}.json"), "w") as file:
            json.dump(segmentation_results.to_json(), file)


if __name__ == "__main__":
    main()
