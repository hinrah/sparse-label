import numpy as np

from constants import Labels
from scipy.spatial import cKDTree

from cross_section import ContourDoesNotExistError
from cross_section_centerline import CrossSectionCenterline


class LabelCrossSection:
    def __init__(self, distance_threshold, with_wall, cross_section, centerline, mask):
        self._distance_threshold = distance_threshold
        self._with_wall = with_wall
        self._cross_section = cross_section
        self._mask = mask
        self._centerline = CrossSectionCenterline(centerline, cross_section)
        self.__label_idx = None
        self.__labels = None
        self.__label_points = None
        self.__potential_foreground_idx = None

    def apply(self):
        try:
            self._label_foreground()
            self._label_background()
            self._mask.set_sparse_mask(self._label_idx, self._labels)
        except ContourDoesNotExistError:
            return

    def _label_foreground(self):
        potential_foreground_points = self._label_points[self._potential_foreground_idx]
        if potential_foreground_points.size == 0:
            raise ContourDoesNotExistError
        if self._with_wall:
            self._labels[self._potential_foreground_idx] = np.where(self._cross_section.projected_inside_wall(potential_foreground_points), Labels.WALL, self._labels[self._potential_foreground_idx])
        self._labels[self._potential_foreground_idx] = np.where(self._cross_section.projected_inside_lumen(potential_foreground_points), Labels.LUMEN, self._labels[self._potential_foreground_idx])

    def _label_background(self):
        labels_with_background = np.where(self._labels == Labels.UNPROCESSED, Labels.BACKGROUND, self._labels)
        self.__labels = np.where(self._centerline.belong_to_centerline(self._label_points), labels_with_background, self._labels)

    @property
    def _label_idx(self):
        if self.__label_idx is None:
            distance_to_plane = self._cross_section.distance_to_plane(self._mask.voxel_center_points)
            self.__label_idx = np.nonzero(distance_to_plane <= self._distance_threshold)[0]
        return self.__label_idx

    @property
    def _labels(self):
        if self.__labels is None:
            self.__labels = np.ones((self._label_idx.size, 1)) * Labels.UNPROCESSED
        return self.__labels

    @property
    def _label_points(self):
        if self.__label_points is None:
            self.__label_points = self._mask.voxel_center_points[self._label_idx]
        return self.__label_points

    @property
    def _potential_foreground_idx(self):
        if self.__potential_foreground_idx is None:
            max_contour_distance = np.max(np.linalg.norm(self._cross_section.all_contour_points - self._cross_section.plane_center, axis=1)) * 1.1
            point_distance = np.linalg.norm(self._label_points - self._cross_section.plane_center, axis=1)
            self.__potential_foreground_idx = (point_distance < max_contour_distance).nonzero()[0]
        return self.__potential_foreground_idx

class LabelCrossSections:
    def __init__(self, distance_threshold, with_wall=True):
        self._distance_threshold = distance_threshold
        self._with_wall = with_wall

    def apply(self, mask, case):
        for cross_section in case.cross_sections:
            self._label_cross_section(mask, cross_section, case.centerline)

    def _label_cross_section(self, mask, cross_section, centerline):
        strategy = LabelCrossSection(self._distance_threshold, self._with_wall, cross_section, centerline, mask)
        strategy.apply()


class LabelCenterline:
    def __init__(self, radius, label_to_create):
        self._radius = radius
        if label_to_create not in [Labels.LUMEN, Labels.BACKGROUND]:
            raise RuntimeError("This strategy can only create lumen or background labels")
        self._label_to_create = label_to_create

    def apply(self, mask, case):
        edge_points = []
        for start_node, end_node in case.centerline.edges():
            edge_points.extend(case.centerline[start_node][end_node]['skeletons'])

        edge_points = np.array(edge_points)
        edge_points = cKDTree(edge_points)
        distance, _ = edge_points.query(mask.voxel_center_points, distance_upper_bound=self._radius)

        if self._label_to_create == Labels.LUMEN:
            out = np.where(distance < self._radius, Labels.LUMEN, Labels.UNPROCESSED).reshape(-1, 1)
        elif self._label_to_create == Labels.BACKGROUND:
            out = np.where(distance > self._radius, Labels.BACKGROUND, Labels.UNPROCESSED).reshape(-1, 1)

        mask.set_mask(out)
