import numpy as np
from scipy.spatial import cKDTree

from edge import Edge


class CrossSectionCenterline:
    def __init__(self, centerline, cross_section):
        self._centerline = centerline
        self._cross_section = cross_section
        self.__cross_section_skeletons = None
        self.__distant_skeletons = None

    def edges(self):
        for start, end in self._centerline.edges():
            yield Edge(np.array(self._centerline[start][end]['skeletons']))

    def belong_to_centerline(self, points):
        return (self._cross_section_skeletons.query(points)[0] < self._distant_skeletons.query(points)[0]).reshape(-1, 1)

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
            if edge.intersects(self._cross_section):
                cross_section_skeletons.append(edge.skeletons)
            else:
                distant_skeletons.extend(edge.skeletons)
        self.__cross_section_skeletons = cKDTree(np.vstack(cross_section_skeletons))
        self.__distant_skeletons = cKDTree(np.vstack(distant_skeletons))
