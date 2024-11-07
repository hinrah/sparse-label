from multiprocessing import Pool

from tqdm import tqdm

import numpy as np

from case import CaseLoader
from label_creator import DefaultLabelCreator

import argparse
from copy import deepcopy


class Processor:
    def __init__(self, label_creator, case_loader):
        self._case_loader = case_loader
        self._label_creator = label_creator

    def _process_one_item_parallel(self, case):
        label_creator = deepcopy(self._label_creator)
        label_creator.create_label(case)

    def process(self):
        for i, case in tqdm(enumerate(self._case_loader)):
            self._label_creator.create_label(case)

    def process_parallel(self, num_threads=4):
        with Pool(processes=num_threads) as pool, tqdm(total=len(self._case_loader)) as pbar:
            for _ in pool.imap(self._process_one_item_parallel, self._case_loader):
                pbar.update()
                pbar.refresh()


def get_max_voxel_size(cases):
    max_voxel_size = 0
    for case in cases:
        max_voxel_size = max(max_voxel_size, max(case.voxel_size))
    return max_voxel_size


def get_min_lumen_centerline_distance(cases):
    min_lumen_centerline_distances = []
    for case in cases:
        try:
            min_lumen_centerline_distances.append(case.min_lumen_centerline_distance())
        except ValueError:
            continue
    return np.percentile(min_lumen_centerline_distances, 5)


def get_max_contour_centerline_distance(cases):
    max_contour_centerline_distances = []
    for case in cases:
        try:
            max_contour_centerline_distances.append(case.max_contour_centerline_distance())
        except ValueError:
            continue
    return max(max_contour_centerline_distances)


def create_sparse_label():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    parser.add_argument('-t', type=float, help="[OPTIONAL] thickness the cross-section annotation. e.g. for a thickness of 1 points that are 0.5mm away from the"
                                   " cross-section plane are considered part of the cross_section. If this is not set the maximum voxel size is used.")
    parser.add_argument('-cr', type=float, help="[OPTIONAL] radius of the lumen voxels that are created based on the centerline. If this is not set it is calculated "
                                    "as the 5th percentile of the distance of all lumen contours to the centerline. It is at leaset the max_voxel_size.")
    parser.add_argument('-vr', type=float, help="[OPTIONAL] vessel radius. Everything that is further away from the centerline is considered as background. If this is "
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
        args.vr = get_max_contour_centerline_distance(case_loader)*1.2

    label_creator = DefaultLabelCreator(args.t, args.cr, args.vr)
    processor = Processor(label_creator, case_loader)
    if args.n > 1:
        processor.process_parallel(args.n)
    else:
        processor.process()

if __name__ == '__main__':
    create_sparse_label()
