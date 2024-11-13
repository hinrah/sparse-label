import os

import numpy as np
import pandas as pd

from evaluation.evaluate3DSegmentationOnSparse import evaluate_segmentations


def save_results_to_csv(path_to_save, metrics):
    metrics_data = {
        "name": [metric[0] for metric in metrics],
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
    }
    df = pd.DataFrame(metrics_data)
    df.to_csv(path_to_save)


if __name__ == "__main__":
    path_to_save = "F:\\sparse_label_dataset\\Dataset002\\results"

    metrics = []
    segmentation_results = evaluate_segmentations()

    metrics.append(("test", segmentation_results))

    save_results_to_csv(os.path.join(path_to_save, "results.csv"), metrics)
