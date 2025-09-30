import numpy as np
from scipy.spatial import cKDTree
from sparselabel.data_handlers.edge import Edge


class CrossSectionScopeClassifier:
    def __init__(self, centerline, cross_section):
        self._centerline = centerline
        self._cross_section = cross_section
        self.__cross_section_skeletons = None
        self.__distant_skeletons = None

    def are_points_within_cross_section_scope(self, points, radius=np.inf):
        are_close_to_centerline = self._cross_section_skeletons.query(points)[0] <= radius
        belong_to_annotated_vessel_branch = self._cross_section_skeletons.query(points)[0] < self._distant_skeletons.query(points)[0]
        return (np.bitwise_and(belong_to_annotated_vessel_branch, are_close_to_centerline)).reshape(-1, 1)

    @property
    def _cross_section_skeletons(self):
        if self.__cross_section_skeletons is None:
            self._classify_skeletons()
        return self.__cross_section_skeletons

    @property
    def _distant_skeletons(self):
        if self.__distant_skeletons is None:
            self._classify_skeletons()
        return self.__distant_skeletons

    def _classify_skeletons(self):
        cross_section_skeletons = [np.zeros((0, 3))]
        distant_skeletons = [np.zeros((0, 3))]

        for edge in self.edges():
            intersections = edge.intersections(self._cross_section)
            in_lumen, out_lumen = self._are_intersection_inside_lumen(intersections)

            if in_lumen and not out_lumen:
                cross_section_skeletons.extend(edge.skeletons)
            elif not in_lumen:
                distant_skeletons.extend(edge.skeletons)
            else:
                self._classify_mixed_skeletons(edge, in_lumen, out_lumen, cross_section_skeletons, distant_skeletons)

        self.__cross_section_skeletons = cKDTree(np.vstack(cross_section_skeletons))
        self.__distant_skeletons = cKDTree(np.vstack(distant_skeletons))

    def _are_intersection_inside_lumen(self, intersections):
        in_lumen, out_lumen = [], []
        if intersections:
            for point in intersections:
                if self._cross_section.is_projected_inside_lumen(point):
                    in_lumen.append(point)
                else:
                    out_lumen.append(point)
        return in_lumen, out_lumen

    def _classify_mixed_skeletons(self, edge, in_lumen, out_lumen, cross_section_skeletons, distant_skeletons):
        in_lumen_tree = cKDTree(np.vstack(in_lumen))
        out_lumen_tree = cKDTree(np.vstack(out_lumen))
        distance_in_lumen, _ = in_lumen_tree.query(edge.skeletons)
        distance_out_lumen, _ = out_lumen_tree.query(edge.skeletons)

        distant_skeletons.extend(
            edge.skeletons[np.nonzero(distance_out_lumen < distance_in_lumen)]
        )
        cross_section_skeletons.extend(
            edge.skeletons[np.nonzero(distance_out_lumen >= distance_in_lumen)]
        )

    def edges(self):
        for start, end in self._centerline.edges():
            yield Edge(np.array(self._centerline[start][end]['skeletons']))

    def has_valid_centerline(self):
        return self._cross_section_skeletons.n > 0
