import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from sparselabel.data_handlers.cross_section import CrossSection
from sparselabel.data_handlers.edge import Edge


class CrossSectionScopeClassifier:
    def __init__(self, centerline: nx.Graph, cross_section: CrossSection) -> None:
        self._centerline = centerline
        self._cross_section = cross_section
        self.__cross_section_skeletons = None
        self.__distant_skeletons = None

    def are_points_within_cross_section_scope(self, points: np.ndarray, radius) -> np.ndarray:
        are_close_to_centerline = self._cross_section_skeletons.query(points)[0] <= radius
        belong_to_annotated_vessel_branch = self._cross_section_skeletons.query(points)[0] < self._distant_skeletons.query(points)[0]
        return (np.bitwise_and(belong_to_annotated_vessel_branch, are_close_to_centerline)).reshape(-1, 1)

    @property
    def _cross_section_skeletons(self) -> cKDTree:
        if self.__cross_section_skeletons is None:
            self._classify_skeletons()
        return self.__cross_section_skeletons

    @property
    def _distant_skeletons(self) -> cKDTree:
        if self.__distant_skeletons is None:
            self._classify_skeletons()
        return self.__distant_skeletons

    def _classify_skeletons(self) -> None:
        cross_section_skeletons = [np.zeros((0, 3))]
        distant_skeletons = [np.zeros((0, 3))]

        for edge in self.edges():
            intersections = edge.intersections(self._cross_section)
            intersections_inside_lumen = self._are_intersection_inside_lumen(intersections)
            if not intersections or np.sum(intersections_inside_lumen) == 0:
                distant_skeletons.extend(edge.skeletons)
            elif np.all(intersections_inside_lumen):
                cross_section_skeletons.extend(edge.skeletons)
            else:  # edge is curved and intersects the plane inside and outside the lumen, e.g. aorta with an axial plane
                in_lumen_intersections = list(np.array(intersections)[intersections_inside_lumen])
                out_lumen_intersections = list(np.array(intersections)[~intersections_inside_lumen])
                distance_to_in_lumen = self._distance_pointcloud_to_points(edge.skeletons, in_lumen_intersections)
                distance_to_out_lumen = self._distance_pointcloud_to_points(edge.skeletons, out_lumen_intersections)

                distant_skeletons.extend(edge.skeletons[np.nonzero(distance_to_out_lumen < distance_to_in_lumen)])
                cross_section_skeletons.extend(edge.skeletons[np.nonzero(distance_to_out_lumen >= distance_to_in_lumen)])

        self.__cross_section_skeletons = cKDTree(np.vstack(cross_section_skeletons))
        self.__distant_skeletons = cKDTree(np.vstack(distant_skeletons))

    def _distance_pointcloud_to_points(self, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
        search_tree = cKDTree(np.vstack(points_b))
        distance, _ = search_tree.query(points_a)
        return distance

    def _are_intersection_inside_lumen(self, intersections: list[np.ndarray]) -> np.ndarray:
        is_in_lumen = []
        for point in intersections:
            if self._cross_section.is_projected_inside_lumen(point):
                is_in_lumen.append(True)
            else:
                is_in_lumen.append(False)
        return np.array(is_in_lumen)

    def edges(self) -> list[Edge]:
        for start, end in self._centerline.edges():
            yield Edge(np.array(self._centerline[start][end]['skeletons']))

    def has_valid_centerline(self):
        return self._cross_section_skeletons.n > 0
