import numpy as np

from sparselabel.data_handlers.cross_section import CrossSection


class Edge:
    """
        Represents the edge between to nodes a centerline graph.

        Attributes:
            skeletons (numpy.ndarray): Ordered point on the path of the edge.
    """
    def __init__(self, skeletons: np.ndarray):
        self.skeletons = skeletons

    @property
    def _skeleton_sampling_distance(self):
        return np.max(np.linalg.norm(self.skeletons[:-1] - self.skeletons[1:], axis=1))

    def intersections(self, cross_section: CrossSection):
        distances = cross_section.distance_to_plane(self.skeletons)
        plane_points = self.skeletons[np.nonzero(distances < self._skeleton_sampling_distance)[0]]
        return list(plane_points)
