import json
import os
from glob import glob

import nibabel as nib
import numpy as np
from networkx.readwrite import json_graph
from scipy.spatial import cKDTree

from constants import Contours, Endings, data_raw, Folders
from cross_section import CrossSection


class Case:
    def __init__(self, case_id, dataset):
        self.case_id = case_id
        self.dataset = dataset
        self.image = None
        self.cross_sections = []
        self.centerline = None

    def load(self):
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
        file_name = self.case_id + Endings.CHANNEL_ZERO + Endings.NIFTI
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
        center_points = np.vstack([cross_section.plane_center for cross_section in self.cross_sections])
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
            case = Case(case_id, self._dataset)
            case.load()
            return case
        else:
            raise StopIteration

    def __len__(self):
        return len(self.case_ids)
