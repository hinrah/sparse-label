from case import CaseLoader
from dataset_characteristics_extraction import get_max_voxel_size, get_min_lumen_centerline_distance, get_max_contour_centerline_distance
from label_creator import DefaultLabelCreator

import argparse

from processor import Processor


def create_sparse_label():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-t', type=float, help="[OPTIONAL] thickness the cross-section annotation. e.g. for a thickness of 1 points that are 0.5mm away from the"
                                   " cross-section plane are considered part of the cross_section. If this is not set the maximum voxel size is used.")
    parser.add_argument('-cr', type=float, help="[OPTIONAL] radius of the lumen voxels that are created based on the centerlines. If this is not set it is calculated "
                                    "as the 5th percentile of the distance of all lumen contours to the centerlines. It is at leaset the max_voxel_size.")
    parser.add_argument('-vr', type=float, help="[OPTIONAL] vessel radius. Everything that is further away from the centerlines is considered as background. If this is "
                                    "not set, it is calculated as 1.2 times the maximum distance of all contour points to there cross-section center")
    parser.add_argument('-n', default=1, type=int, help="[OPTIONAL] number of processes. If not set this will run synchronous on one process.")
    args = parser.parse_args()

    case_loader = CaseLoader(args.d)

    max_voxel_size = get_max_voxel_size(case_loader)
    if not args.t:
        args.t = max_voxel_size

    if not args.cr:
        args.cr = max(get_min_lumen_centerline_distance(case_loader), max_voxel_size)

    if not args.vr:
        args.vr = get_max_contour_centerline_distance(case_loader) * 1.2

    label_creator = DefaultLabelCreator(args.t, args.cr, args.vr)
    processor = Processor(label_creator, case_loader)
    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()

if __name__ == '__main__':
    create_sparse_label()
