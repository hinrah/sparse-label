import numpy as np


class Edge:
    def __init__(self, skeletons):
        self.skeletons = skeletons

    @property
    def _skeleton_sampling_distance(self):
        return np.max(np.linalg.norm(self.skeletons[:-1] - self.skeletons[1:], axis=1))

    def intersects(self, cross_section):
        distances = cross_section.distance_to_plane(self.skeletons)
        plane_points = self.skeletons[np.nonzero(distances < self._skeleton_sampling_distance)[0]]
        if plane_points.size == 0:
            return False
        for point in plane_points:
            if cross_section.is_projected_inside_lumen(point):
                return True
        return False
