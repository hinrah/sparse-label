import json
import os
from glob import glob
from multiprocessing import Pool

from scipy.spatial import cKDTree
from tqdm import tqdm

import numpy as np
import nibabel as nib

from constants import Labels, data_raw, Folders, Endings, Contours
from cross_section import CrossSection
from labeling_strategies import LabelDistantVoxelsAsUnknown, LabelVoxelsBasedOnProjectionToCrossSection, LabelUnknownCrosssSectionBackground, LabelCenterline
from mask_image import SparseMaskImage

from networkx.readwrite import json_graph
import argparse
from copy import deepcopy


class Case:
    def __init__(self, case_id, dataset):
        self.case_id = case_id
        self.dataset = dataset
        self.image = None
        self.cross_sections = []
        self.centerline = None
        self._load()

    def _load(self):
        self._load_image()
        self._load_cross_sections()
        self._load_centerline()

    def _load_cross_sections(self):
        raw_cross_sections = self._load_raw_cross_sections()
        self.cross_sections = []
        for raw_cross_section in raw_cross_sections.values():
            if not Contours.INNER in raw_cross_section.keys() or not Contours.OUTER in raw_cross_section.keys():
                print("Cross-section does not have both contours and is ignored.")
                continue
            inner_contour_points = np.array(raw_cross_section[Contours.INNER])
            outer_contour_points = np.array(raw_cross_section[Contours.OUTER])
            self.cross_sections.append(CrossSection(inner_contour_points, outer_contour_points))

    def _load_raw_cross_sections(self):
        file_name = self.case_id + Endings.JSON
        contour_path = os.path.join(data_raw, self.dataset, Folders.CONTOURS, file_name)
        with open(contour_path, "r") as file:
            contours = json.load(file)
        return contours

    def _load_image(self):
        file_name = self.case_id + "_0000" + Endings.NIFTI
        image_path = os.path.join(data_raw, self.dataset, Folders.IMAGES, file_name)
        self.image = nib.load(image_path)

    def _load_centerline(self):
        file_name = self.case_id + Endings.JSON
        centerline_path = os.path.join(data_raw, self.dataset, Folders.CENTERLINES, file_name)
        with open(centerline_path, "r") as file:
            centerline_raw = json.load(file)
        self.centerline = json_graph.node_link_graph(centerline_raw, link="edges")

    @property
    def image_shape(self):
        return self.image.shape

    @property
    def affine(self):
        return self.image.affine

    @property
    def voxel_size(self):
        return self.image.header['pixdim'][1:4]

    def min_lumen_centerline_distance(self):
        centerline_points = self._all_centerline_points()
        lumen_points = self._all_lumen_points()
        if not centerline_points.size or not lumen_points.size:
            raise ValueError()
        centerline_tree = cKDTree(centerline_points)
        distances, _ = centerline_tree.query(lumen_points, k=1)
        return min(distances)

    def max_contour_centerline_distance(self):
        center_points = np.vstack([cross_section.plane_center for cross_section in self.cross_sections])# = self._all_centerline_points()
        contour_points = self._all_contour_points()

        if not center_points.size or not contour_points.size:
            raise ValueError
        centerline_tree = cKDTree(center_points)
        distances, _ = centerline_tree.query(contour_points, k=1)
        return max(distances)

    def _all_contour_points(self):
        raw_cross_sections = self._load_raw_cross_sections()
        contour_points = [np.zeros((0, 3))]
        for raw_cross_section in raw_cross_sections.values():
            if not Contours.INNER in raw_cross_section.keys() or not Contours.OUTER in raw_cross_section.keys():
                print("Cross-section does not have both contours and is ignored.")
                continue
            contour_points.append(np.array(raw_cross_section[Contours.INNER]))
            contour_points.append(np.array(raw_cross_section[Contours.OUTER]))
        return np.vstack(contour_points)

    def _all_lumen_points(self):
        raw_cross_sections = self._load_raw_cross_sections()
        lumen_points = [np.zeros((0, 3))]
        for raw_cross_section in raw_cross_sections.values():
            if not Contours.INNER in raw_cross_section.keys():
                continue
            lumen_points.append(np.array(raw_cross_section[Contours.INNER]))
        return np.vstack(lumen_points)

    def _all_centerline_points(self):
        centerline_points = [np.zeros((0, 3))]
        for start, end in self.centerline.edges():
            centerline_points.append(self.centerline[start][end]['skeletons'])
        return np.vstack(centerline_points)

    def save_label(self, voxel_mask):
        file_name = self.case_id + Endings.NIFTI
        label_path = os.path.join(data_raw, self.dataset, Folders.LABELS, file_name)
        out_image = nib.Nifti1Image(np.astype(voxel_mask, np.int16), self.affine)
        nib.save(out_image, label_path)


class DefaultLabelCreator:
    def __init__(self, cross_section_thickness, min_lumen_radius, max_wall_radius):
        self._cross_section_masks = []
        self._joined_mask = None
        self._cross_section_thickness = cross_section_thickness
        self._min_lumen_radius = min_lumen_radius
        self._max_wall_radius = max_wall_radius

    def _reset(self):
        self._cross_section_masks = []
        self._joined_mask = None

    def create_label(self, case):
        self._reset()
        self._apply_cross_section_based_strategies(case)
        self._join_cross_section_masks(case)
        self._apply_centerline_based_strategies(case)
        case.save_label(self._joined_mask.mask)

    def _apply_cross_section_based_strategies(self, case):
        for cross_section in case.cross_sections:
            mask = SparseMaskImage(case.image_shape, case.affine)

            label_non_plane_voxels = LabelDistantVoxelsAsUnknown(cross_section, self._cross_section_thickness/2)
            label_in_plane_voxels = LabelVoxelsBasedOnProjectionToCrossSection(cross_section)
            label_background_as_unknown_based_on_centerline = LabelUnknownCrosssSectionBackground(cross_section, case.centerline)
            mask.label_points_with_label(label_non_plane_voxels, Labels.UNPROCESSED)
            mask.label_points_with_label(label_in_plane_voxels, Labels.UNPROCESSED)
            mask.label_points_with_label(label_background_as_unknown_based_on_centerline, Labels.BACKGROUND)

            self._cross_section_masks.append(mask)

    def _join_cross_section_masks(self, case):
        if len(self._cross_section_masks) == 0:
            self._joined_mask = SparseMaskImage(case.image_shape, case.affine)
            self._joined_mask.set_mask(np.ones(case.image_shape)*Labels.UNKNOWN)
            print("Case without a cross-section. This will give limited additional info as only centerline based labels are created. Consider removing this case.")
            return

        all_masks = np.stack([mask.mask for mask in self._cross_section_masks])

        min_label = np.min(all_masks, axis=0)
        is_valid = np.all(np.logical_or(all_masks == min_label, all_masks == Labels.UNKNOWN), axis=0)

        joined_image = np.where(is_valid, min_label, Labels.UNKNOWN)

        self._joined_mask = SparseMaskImage(case.image_shape, case.affine)
        self._joined_mask.set_mask(joined_image)

    def _apply_centerline_based_strategies(self, case):
        label_lumen_based_on_centerline = LabelCenterline(case.centerline, self._min_lumen_radius, Labels.LUMEN)
        label_background_based_on_centerline = LabelCenterline(case.centerline, self._max_wall_radius, Labels.BACKGROUND)
        self._joined_mask.label_points_with_label(label_lumen_based_on_centerline, Labels.UNKNOWN)
        self._joined_mask.label_points_with_label(label_background_based_on_centerline, Labels.UNKNOWN)

class CaseLoader:
    def __init__(self, dataset):
        self.case_ids = []
        self._dataset = dataset
        self._search_cases()
        self.index = 0

    def _search_cases(self):
        self.case_ids = []
        file_name_search = "*" + Endings.CHANNEL_ZERO + Endings.NIFTI
        for path in glob(os.path.join(data_raw, self._dataset, Folders.IMAGES, file_name_search)):
            case_id = os.path.basename(path)[:-len(Endings.CHANNEL_ZERO) - len(Endings.NIFTI)]
            self.case_ids.append(case_id)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.case_ids):
            case_id = self.case_ids[self.index]
            self.index += 1
            return Case(case_id, self._dataset)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.case_ids)

class Processor:
    def __init__(self, label_creator, case_loader):
        self._case_loader = case_loader
        self._label_creator = label_creator

    #def _search_cases(self):
    #    self._case_ids = []
    #    file_name_search = "*" + Endings.CHANNEL_ZERO + Endings.NIFTI
    #    for path in glob(os.path.join(data_raw, self._dataset, Folders.IMAGES, file_name_search)):
    #        case_id = os.path.basename(path)[:-len(Endings.CHANNEL_ZERO) - len(Endings.NIFTI)]
    #        self._case_ids.append(case_id)

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
