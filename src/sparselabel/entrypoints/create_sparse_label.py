from sparselabel.dataset_checks.dataset_checks import CenterlineInsideInnerContour, InnerContourWithinOuter
from sparselabel.data_handlers.case import Case
from sparselabel.case_loader import CaseLoader
from sparselabel.dataset_characteristics_extraction import get_max_voxel_size, get_min_lumen_centerline_distance, get_max_contour_centerline_distance
from sparselabel.dataset_config import DatasetConfig
from sparselabel.dataset_checks.dataset_tester import DatasetTester
from sparselabel.label_strategies.label_creator import LabelCreator
from sparselabel.logging_config import logger

import argparse

from sparselabel.label_strategies.labeling_strategies import LabelCenterline, LabelCrossSections, LabelEndingCrossSections
from sparselabel.processor import Processor


def create_sparse_label():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-t', type=float,
                        help="[OPTIONAL] thickness the cross-section annotation. e.g. for a thickness of 1 points that are 0.5mm away from the"
                             " cross-section plane are considered part of the cross_section. If this is not set the maximum voxel size is used.")
    parser.add_argument('-cr', type=float,
                        help="[OPTIONAL] radius of the lumen voxels that are created based on the centerlines. If this is not set it is calculated "
                             "as the 5th percentile of the distance of all lumen contoursTr to the centerlines. It is at leaset the max_voxel_size.")
    parser.add_argument('-br', type=float,
                        help="[OPTIONAL] background radius for cross-sections. Radius around the cross-section centerpoint that is considered as background.")
    parser.add_argument('-vr', type=float,
                        help="[OPTIONAL] vessel radius. Everything that is further away from the centerlines is considered as background. If this is "
                             "not set, it is calculated as 1.2 times the maximum distance of all contour points to there cross-section center")
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    parser.add_argument('--ending', action=argparse.BooleanOptionalAction, default=False,
                        help="[OPTIONAL] If the vessel has a cross-section annotation that is known to be placed at the end of the vessel section that should"
                             " be segmented.")
    parser.add_argument('--wall', action=argparse.BooleanOptionalAction,
                        help="[OPTIONAL] Should the labels be created only for the vessel lumen or for a lumen and a surrounding wall.")
    parser.add_argument('--checkDataset', action=argparse.BooleanOptionalAction, default=False,
                        help="[OPTIONAL] Run basic checks on the validity of the dataset before creating the labels.")
    args = parser.parse_args()

    if args.checkDataset:
        check_dataset(args)

    dataset_config = DatasetConfig(args.d)

    case_loader = CaseLoader(dataset_config, Case)

    max_voxel_size = None

    print(1)
    if not args.t:
        if max_voxel_size is None:
            max_voxel_size = get_max_voxel_size(case_loader)
        args.t = max_voxel_size

    if args.cr is None:
        if max_voxel_size is None:
            max_voxel_size = get_max_voxel_size(case_loader)
        args.cr = max(get_min_lumen_centerline_distance(case_loader), max_voxel_size)

    if not args.vr:
        args.vr = get_max_contour_centerline_distance(case_loader) * 1.2

    if not args.br:
        args.br = args.vr * 1.2

    strategies = [LabelCrossSections(dataset_config, args.t / 2, with_wall=args.wall, radius=args.br),
                  LabelCenterline(dataset_config, args.cr, dataset_config.lumen_value),
                  LabelCenterline(dataset_config, args.vr, dataset_config.background_value)]

    if args.ending:
        strategies.append(LabelEndingCrossSections(dataset_config, args.t / 2, radius=args.br))

    label_creator = LabelCreator(strategies, dataset_config)
    processor = Processor(label_creator, case_loader)
    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()


def check_dataset(args):
    logger.setLevel("INFO")

    tests = [CenterlineInsideInnerContour(), InnerContourWithinOuter()]
    dataset_config = DatasetConfig(args.d)
    case_loader = CaseLoader(dataset_config, Case)

    tester = DatasetTester(tests)
    processor = Processor(tester, case_loader)

    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()

    logger.info("If you do not see any error messages above, the dataset passed the validity checks")


if __name__ == '__main__':
    create_sparse_label()
