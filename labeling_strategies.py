import numpy as np
from networkx import Graph

from constants import Labels
from scipy.spatial import cKDTree

class LabelDistantVoxelsAsUnknown:
    def __init__(self, contour, distance_threshold):
        self._contour = contour
        self._distance_threshold = distance_threshold

    def apply(self, points):
        label = np.ones((points.shape[0], 1))*Labels.UNPROCESSED
        distance_to_plane = self._contour.distance_to_plane(points)
        label[distance_to_plane > self._distance_threshold] = Labels.UNKNOWN
        return label


class LabelVoxelsBasedOnProjectionToCrossSection:
    def __init__(self, cross_section):
        self._cross_section = cross_section

    def apply(self, points):
        labels = np.zeros((points.shape[0], 1))
        for i, point in enumerate(points):
            if self._cross_section.is_projected_inside_lumen(point):
                labels[i] = Labels.LUMEN
            elif self._cross_section.is_projected_inside_wall(point):
                labels[i] = Labels.WALL
            else:
                labels[i] = Labels.BACKGROUND
        return labels


class LabelUnknownCrosssSectionBackground:
    def __init__(self, cross_section, centerline: Graph):
        self._cross_section = cross_section
        self._centerline = centerline

        self._initialize_edge_points()

    def _initialize_edge_points(self):
        good_edge_points = []
        bad_edge_points = []
        for start_node, end_node in self._centerline.edges():
            edge_points = np.array(self._centerline[start_node][end_node]['skeletons'])
            edge_points[:, :2] *= -1
            max_edgepoint_distance = np.max(np.linalg.norm(edge_points[:-1]-edge_points[1:], axis=1))
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

    def apply(self, points):
        good_distances, _ = self._good_edge_points.query(points)
        bad_distances, _ = self._bad_edge_points.query(points)
        return np.where(good_distances<bad_distances, Labels.BACKGROUND, Labels.UNKNOWN).reshape(-1,1)

class LabelCenterline:
    def __init__(self, centerline, radius, label_to_create):
        self._centerline = centerline
        self._radius = radius
        if label_to_create not in [Labels.LUMEN, Labels.BACKGROUND]:
            raise RuntimeError("This strategy can only create lumen or background labels")
        self._label_to_create = label_to_create

    def apply(self, points):
        edge_points = []
        for start_node, end_node in self._centerline.edges():
            edge_points.extend(self._centerline[start_node][end_node]['skeletons'])

        edge_points = np.array(edge_points)
        edge_points[:, :2] *= -1
        edge_points = cKDTree(edge_points)
        distance, _ = edge_points.query(points)

        if self._label_to_create == Labels.LUMEN:
            return np.where(distance < self._radius, Labels.LUMEN, Labels.UNKNOWN).reshape(-1, 1)
        elif self._label_to_create == Labels.BACKGROUND:
            return np.where(distance > self._radius, Labels.BACKGROUND, Labels.UNKNOWN).reshape(-1, 1)
