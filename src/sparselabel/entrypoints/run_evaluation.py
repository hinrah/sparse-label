import argparse
import json
import os

from sparselabel.constants import ENCODING
from sparselabel.dataset_config import DatasetConfig
from sparselabel.evaluation.evaluate3DSegmentationOnSparse import evaluate_segmentations
from sparselabel.evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn3DLabel, SegmentationEvaluator2DContourOn2DCrossSections, \
    SegmentationEvaluatorAllContoursOn3DLabel
from sparselabel.evaluation.parameter_extractor import ParameterExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    parser.add_argument('-d', nargs='+', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-e', help="[REQUIRED] evaluator to use [2D, 3D, 3D_complete_case, 2D_QuantitativeParameters]. ")
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
            complete_case = False
            size = int(args.fow_2d // args.res_2d)
            evaluator = SegmentationEvaluator2DContourOn2DCrossSections(dataset_config, (args.res_2d, args.res_2d), (size, size))
        elif args.e == "3D":
            complete_case = False
            evaluator = SegmentationEvaluator2DContourOn3DLabel(dataset_config)
        elif args.e == "3D_complete_case":
            complete_case = True
            evaluator = SegmentationEvaluatorAllContoursOn3DLabel(dataset_config)
        elif args.e == "2D_QuantitativeParameters":
            complete_case = False
            size = int(args.fow_2d // args.res_2d)
            evaluator = ParameterExtractor(dataset_config, (args.res_2d, args.res_2d), (size, size))
        else:
            raise RuntimeError(f"Evaluator {args.e} is not implemented.")
        segmentation_results = evaluate_segmentations(dataset_config, args.n, evaluator, complete_case)
        with open(os.path.join(dataset_config.prediction_path, f"evaluation_results_{args.e}.json"), "w", encoding=ENCODING) as file:
            json.dump(segmentation_results.to_json(), file)


if __name__ == "__main__":
    main()
