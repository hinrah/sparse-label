import numpy as np
from scipy.spatial import cKDTree

from sparselabel.data_handlers.edge import Edge


class CrossSectionCenterline:
    def __init__(self, centerline, cross_section):
        self._centerline = centerline
        self._cross_section = cross_section
        self.__cross_section_skeletons = None
        self.__distant_skeletons = None

    def edges(self):
        for start, end in self._centerline.edges():
            yield Edge(np.array(self._centerline[start][end]['skeletons']))

    def belong_to_centerline(self, points, radius=np.inf):
        return (np.bitwise_and(self._cross_section_skeletons.query(points)[0] < self._distant_skeletons.query(points)[0],
                               self._cross_section_skeletons.query(points)[0] <= radius)).reshape(-1, 1)

    def has_valid_centerline(self):
        return self._cross_section_skeletons.n > 0

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
            in_lumen, out_lumen = edge.intersections(self._cross_section)
            if in_lumen and not out_lumen:
                cross_section_skeletons.extend(edge.skeletons)
            elif not in_lumen:
                distant_skeletons.extend(edge.skeletons)
            else:
                in_lumen = cKDTree(np.vstack(in_lumen))
                out_lumen = cKDTree(np.vstack(out_lumen))
                distance_in_lumen, _ = in_lumen.query(edge.skeletons)
                distance_out_lumen, _ = out_lumen.query(edge.skeletons)
                distant_skeletons.extend(edge.skeletons[np.nonzero(distance_out_lumen < distance_in_lumen)])
                cross_section_skeletons.extend(edge.skeletons[np.nonzero(distance_out_lumen >= distance_in_lumen)])
        self.__cross_section_skeletons = cKDTree(np.vstack(cross_section_skeletons))
        self.__distant_skeletons = cKDTree(np.vstack(distant_skeletons))
