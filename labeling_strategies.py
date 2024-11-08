import numpy as np
from networkx import Graph

from constants import Labels
from scipy.spatial import cKDTree


class LabelCrossSections:
    def __init__(self, _distance_threshold):
        self._distance_threshold = _distance_threshold

    def apply(self, mask, case):
        for cross_section in case.cross_sections:
            label_cross_section = LabelCrossSection(cross_section, self._distance_threshold, case.centerline)
            label_cross_section.apply(mask)



class LabelCrossSection:
    def __init__(self, cross_section, distance_threshold, centerline):
        self._cross_section = cross_section
        self._distance_threshold = distance_threshold
        self._centerline = centerline
        self._initialize_edge_points()

    def _initialize_edge_points(self):
        good_edge_points = [np.zeros((0, 3))]
        bad_edge_points = [np.zeros((0, 3))]
        for start_node, end_node in self._centerline.edges():
            edge_points = np.array(self._centerline[start_node][end_node]['skeletons'])
            max_edgepoint_distance = np.max(np.linalg.norm(edge_points[:-1] - edge_points[1:], axis=1))
            distances = self._cross_section.distance_to_plane(edge_points)
            plane_points = edge_points[np.nonzero(distances < max_edgepoint_distance)[0]]
            is_edge = plane_points.size > 0
            for point in plane_points:
                if not self._cross_section.is_projected_inside_lumen(point):
                    is_edge = False
            if is_edge:
                good_edge_points.append(edge_points)
            else:
                bad_edge_points.extend(edge_points)

        self._bad_edge_points = cKDTree(np.vstack(bad_edge_points))
        self._good_edge_points = cKDTree(np.vstack(good_edge_points))

    def apply(self, mask):
        distance_to_plane = self._cross_section.distance_to_plane(mask.voxel_center_points)

        label_idx = np.nonzero(distance_to_plane <= self._distance_threshold)[0]
        labels = np.ones((label_idx.size, 1)) * Labels.BACKGROUND
        label_points = mask.voxel_center_points[label_idx]

        max_contour_distance = np.max(np.linalg.norm(self._cross_section.all_contour_points - self._cross_section.plane_center, axis=1)) * 1.1

        point_distance = np.linalg.norm(label_points - self._cross_section.plane_center, axis=1)
        potential_foreground_idx = (point_distance < max_contour_distance).nonzero()[0]

        for idx in potential_foreground_idx:
            if self._cross_section.is_projected_inside_lumen(label_points[idx]):
                labels[idx] = Labels.LUMEN
            elif self._cross_section.is_projected_inside_wall(label_points[idx]):
                labels[idx] = Labels.WALL


        good_distances = self._good_edge_points.query(label_points)[0].reshape(-1,1)
        bad_distances = self._bad_edge_points.query(label_points)[0].reshape(-1,1)

        labels_background_unknown = np.where(labels == Labels.BACKGROUND, Labels.UNPROCESSED, labels)

        labels = np.where(good_distances < bad_distances, labels, labels_background_unknown)

        mask.set_sparse_mask(label_idx, labels)

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
        distance, _ = edge_points.query(mask.voxel_center_points)

        if self._label_to_create == Labels.LUMEN:
            out = np.where(distance < self._radius, Labels.LUMEN, Labels.UNPROCESSED).reshape(-1, 1)
        elif self._label_to_create == Labels.BACKGROUND:
            out = np.where(distance > self._radius, Labels.BACKGROUND, Labels.UNPROCESSED).reshape(-1, 1)

        mask.set_mask(out)
