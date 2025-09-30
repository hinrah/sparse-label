from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree

from sparselabel.constants import LabelStrategies
from sparselabel.data_handlers.case import Case
from sparselabel.data_handlers.cross_section import ContourDoesNotExistError
from sparselabel.data_handlers.cross_section_scope_classifier import CrossSectionScopeClassifier
from sparselabel.data_handlers.mask_image import UNPROCESSED, SparseMaskImage


class LabelingStrategy(ABC):

    @abstractmethod
    def apply(self, mask: SparseMaskImage, case: Case):
        pass


class LabelCrossSections(LabelingStrategy):
    def __init__(self, dataset_config, distance_threshold, with_wall=True, radius=np.inf):
        self._dataset_config = dataset_config
        self._distance_threshold = distance_threshold
        self._with_wall = with_wall
        self._radius = radius

    def apply(self, mask: SparseMaskImage, case: Case):
        for cross_section in case.cross_sections:
            self._label_cross_section(mask, cross_section, case.centerline)

    def _label_cross_section(self, mask, cross_section, centerline):
        strategy = LabelCrossSection(self._dataset_config, self._distance_threshold, self._with_wall, cross_section, centerline, mask, self._radius)
        strategy.apply()


class LabelCrossSection:
    def __init__(self, dataset_config, distance_threshold, with_wall, cross_section, centerline, mask, radius):
        self._dataset_config = dataset_config
        self._distance_threshold = distance_threshold
        self._with_wall = with_wall
        self._cross_section = cross_section
        self._mask = mask
        self._centerline = CrossSectionScopeClassifier(centerline, cross_section)
        self.__considered_voxel_indices = None
        self.__labels = None
        self.__label_points = None
        self.__potential_foreground_idx = None
        self._radius = radius

    def apply(self):
        try:
            self._label_foreground()
            self._label_background()
            self._mask.set_sparse_mask(self._considered_voxel_indices, self._labels)
        except ContourDoesNotExistError:
            pass

    def _label_foreground(self):
        potential_foreground_points = self._label_points[self._potential_foreground_idx]
        if potential_foreground_points.size == 0:
            raise ContourDoesNotExistError
        if self._with_wall:
            self._labels[self._potential_foreground_idx] = np.where(self._cross_section.projected_inside_wall(potential_foreground_points),
                                                                    self._dataset_config.wall_value, self._labels[self._potential_foreground_idx])
        self._labels[self._potential_foreground_idx] = np.where(self._cross_section.projected_inside_lumen(potential_foreground_points),
                                                                self._dataset_config.lumen_value, self._labels[self._potential_foreground_idx])

    def _label_background(self):
        labels_with_background = np.where(self._labels == UNPROCESSED, self._dataset_config.background_value, self._labels)
        self.__labels = np.where(self._centerline.are_points_within_cross_section_scope(self._label_points, radius=np.inf),
                                 labels_with_background, self._labels)

    @property
    def _considered_voxel_indices(self):
        if self.__considered_voxel_indices is None:
            distance_to_plane = self._cross_section.distance_to_plane(self._mask.voxel_center_points)
            is_plane_voxel = distance_to_plane <= self._distance_threshold
            is_relevant_voxel = (np.linalg.norm(self._mask.voxel_center_points - self._cross_section.plane_center, axis=1) <= self._radius).reshape((-1, 1))

            self.__considered_voxel_indices = np.nonzero(np.logical_and(is_plane_voxel, is_relevant_voxel))[0]
        return self.__considered_voxel_indices

    @property
    def _labels(self):
        if self.__labels is None:
            self.__labels = np.ones((self._considered_voxel_indices.size, 1)) * UNPROCESSED
        return self.__labels

    @property
    def _label_points(self):
        if self.__label_points is None:
            self.__label_points = self._mask.voxel_center_points[self._considered_voxel_indices]
        return self.__label_points

    @property
    def _potential_foreground_idx(self):
        if self.__potential_foreground_idx is None:
            max_contour_distance = np.max(
                np.linalg.norm(self._cross_section.all_contour_points - self._cross_section.plane_center, axis=1)) * LabelStrategies.CONSIDERED_AREA_FACTOR
            point_distance = np.linalg.norm(self._label_points - self._cross_section.plane_center, axis=1)
            self.__potential_foreground_idx = (point_distance < max_contour_distance).nonzero()[0]
        return self.__potential_foreground_idx


class LabelCenterline(LabelingStrategy):
    def __init__(self, dataset_config, radius, label_to_create):
        self._dataset_config = dataset_config
        self._radius = radius
        if label_to_create not in [self._dataset_config.lumen_value, self._dataset_config.background_value]:
            raise RuntimeError("This strategy can only create lumen or background labels")
        self._label_to_create = label_to_create

    def apply(self, mask: SparseMaskImage, case: Case):
        centerline_tree = self.create_centerline_distance_tree(case)
        distance, _ = centerline_tree.query(mask.voxel_center_points, distance_upper_bound=self._radius)

        if self._label_to_create == self._dataset_config.lumen_value:
            out = np.where(distance < self._radius, self._dataset_config.lumen_value, UNPROCESSED).reshape(-1, 1)
        elif self._label_to_create == self._dataset_config.background_value:
            out = np.where(distance > self._radius, self._dataset_config.background_value, UNPROCESSED).reshape(-1, 1)
        else:
            raise NotImplementedError

        mask.set_mask(out)

    @staticmethod
    def create_centerline_distance_tree(case):
        edge_points = []
        for start_node, end_node in case.centerline.edges():
            edge_points.extend(case.centerline[start_node][end_node]['skeletons'])
        return cKDTree(edge_points)


class LabelEndingCrossSections(LabelingStrategy):
    def __init__(self, dataset_config, distance_threshold, radius=np.inf):
        self._dataset_config = dataset_config
        self._distance_threshold = distance_threshold
        self._radius = radius

    def apply(self, mask: SparseMaskImage, case: Case):
        for cross_section in case.cross_sections:
            if cross_section.is_ending_cross_section:
                self._label_ending_cross_sections(mask, cross_section)

    def _label_ending_cross_sections(self, mask, cross_section):
        labels = np.ones((mask.voxel_center_points.shape[0], 1)) * UNPROCESSED
        background_voxels = np.nonzero(np.logical_and(np.dot(mask.voxel_center_points - cross_section.plane_center, cross_section.ending_normal) > 0,
                                                      (np.linalg.norm(mask.voxel_center_points - cross_section.plane_center, axis=1) <= self._radius).reshape(
                                                          (-1, 1))))[0]
        labels[background_voxels] = self._dataset_config.background_value
        mask.set_mask(labels)
