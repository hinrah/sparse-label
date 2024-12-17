from check_datasets import CenterlineInsideLumen, LumenInsideWall, NumArteries
from case import CaseLoader, Case
from dataset_characteristics_extraction import get_max_voxel_size, get_min_lumen_centerline_distance, get_max_contour_centerline_distance
from dataset_config import DatasetConfig
from dataset_tester import DatasetTester
from label_creator import LabelCreator
from logging_config import logger

import argparse

from labeling_strategies import LabelCenterline, LabelCrossSections, LabelEndingCrossSections
from processor import Processor


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
    parser.add_argument('--ending', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wall', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_config = DatasetConfig(args.d)

    case_loader = CaseLoader(dataset_config, Case)

    max_voxel_size = get_max_voxel_size(case_loader)
    if not args.t:
        args.t = max_voxel_size

    if args.cr is None:
        args.cr = max(get_min_lumen_centerline_distance(case_loader), max_voxel_size)

    if not args.vr:
        args.vr = get_max_contour_centerline_distance(case_loader) * 1.2

    if not args.br:
        args.br = args.vr*1.2

    strategies = [LabelCrossSections(dataset_config, args.t / 2, with_wall=args.wall, radius=args.br),
                  LabelCenterline(dataset_config, args.cr, dataset_config.lumen_value),
                  LabelCenterline(dataset_config, args.vr, dataset_config.wall_value)]

    if args.ending:
        strategies.append(LabelEndingCrossSections(dataset_config, args.t / 2, radius=args.br))

    label_creator = LabelCreator(strategies, dataset_config)
    processor = Processor(label_creator, case_loader)
    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()


def check_dataset():
    logger.setLevel("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    parser.add_argument('-na', default=1, type=int, help="[OPTIONAL] number of arteries (connectedCenterlineComponents) to expect. Default = 1")
    parser.add_argument('--wall', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    tests = [CenterlineInsideLumen(), LumenInsideWall(), NumArteries(args.na)]
    dataset_config = DatasetConfig(args.d)
    case_loader = CaseLoader(dataset_config, Case)

    tester = DatasetTester(tests)
    processor = Processor(tester, case_loader)

    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()

    logger.info("If you do not see any error messages above, the dataset is valid.")

if __name__ == '__main__':
    #check_dataset()
    create_sparse_label()
