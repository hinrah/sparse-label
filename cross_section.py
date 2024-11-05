import numpy as np
from sklearn.decomposition import PCA

from contour import Contour


class CrossSection:
    def __init__(self, lumen_contour, wall_contour=None):
        self._pca = self._create_pca(lumen_contour, wall_contour)

        self._lumen_contour = Contour(self.transform_points_to_plane_coordinates(lumen_contour))
        if wall_contour is not None:
            self._wall_contour = Contour(self.transform_points_to_plane_coordinates(wall_contour))
            if not self._wall_contour.contains(self._lumen_contour.polygon):
                raise RuntimeError("lumen contour needs to be inside wall contour")
        else:
            self._wall_contour = None

    def _create_pca(self, lumen_contour_points, wall_contour_points):
        if wall_contour_points is None:
            points = lumen_contour_points
        else:
            points = np.vstack((lumen_contour_points, wall_contour_points))
        pca = PCA(n_components=3, svd_solver="full")
        pca.fit(points)
        return pca

    @property
    def plane_normal(self):
        return self._pca.components_[2:3].T

    @property
    def plane_center(self):
        return self._pca.mean_

    @property
    def plane_transform(self):
        return self._pca.components_[:2].T

    def transform_points_to_plane_coordinates(self, points):
        points = points.reshape(-1, 3)
        return (points - self.plane_center) @ self.plane_transform

    def distance_to_plane(self, points):
        return np.abs((points - self.plane_center) @ self.plane_normal)

    def is_projected_inside_lumen(self, point):
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._lumen_contour.contains_point(projected_point)

    def is_projected_inside_wall(self, point):
        if self._wall_contour is None:
            return False
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._wall_contour.contains_point(projected_point)
