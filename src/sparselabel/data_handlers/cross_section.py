import numpy as np
from sklearn.decomposition import PCA
from skimage.draw import polygon  # pylint: disable=no-name-in-module

from sparselabel.data_handlers.contour import Contour


class CrossSection:
    def __init__(self, dataset_config, identifier, lumen_contour, outer_wall_contour=None, ending_normal=None):
        self._dataset_config = dataset_config
        self.identifier = identifier
        self._lumen_contour_points = lumen_contour
        self._outer_wall_contour_points = outer_wall_contour
        self._pca = self._create_pca(lumen_contour, outer_wall_contour)
        self._ending_normal = ending_normal

        self._lumen_contour = Contour(self.transform_points_to_plane_coordinates(lumen_contour))
        if outer_wall_contour is not None:
            self._outer_wall_contour = Contour(self.transform_points_to_plane_coordinates(outer_wall_contour))
        else:
            self._outer_wall_contour = None

    def lumen_is_inside_wall(self):
        if self._outer_wall_contour is None:
            raise ContourDoesNotExistError
        return self._outer_wall_contour.contains(self._lumen_contour.polygon)

    @property
    def all_contour_points(self):
        if self._outer_wall_contour_points is None:
            return self._lumen_contour_points
        return np.vstack((self._lumen_contour_points, self._outer_wall_contour_points))

    @property
    def lumen_points(self):
        return self._lumen_contour_points

    @property
    def outer_wall_points(self):
        return self._outer_wall_contour_points

    def _create_pca(self, lumen_contour_points, outer_wall_contour_points):
        if outer_wall_contour_points is None:
            points = lumen_contour_points
        else:
            points = np.vstack((lumen_contour_points, outer_wall_contour_points))
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

    def plane_x_axis(self):
        return self._pca.components_[0]

    def plane_y_axis(self):
        return self._pca.components_[1]

    def transform_points_to_plane_coordinates(self, points):
        points = points.reshape(-1, 3)
        return (points - self.plane_center) @ self.plane_transform

    def distance_to_plane(self, points):
        return np.abs((points - self.plane_center) @ self.plane_normal)

    def is_projected_inside_lumen(self, point):
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._lumen_contour.contains_point(projected_point)

    def projected_inside_lumen(self, points):
        return np.array([self.is_projected_inside_lumen(point) for point in points]).reshape(-1, 1)

    def projected_inside_wall(self, points):
        return np.array([self.is_projected_inside_wall(point) for point in points]).reshape(-1, 1)

    def is_projected_inside_wall(self, point):
        if self._outer_wall_contour is None:
            raise ContourDoesNotExistError
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._outer_wall_contour.contains_point(projected_point)

    def create_pixel_mask(self, pixel_dims, image_shape):
        mask = np.ones(image_shape, dtype=np.uint8) * self._dataset_config.background_value

        if self._outer_wall_contour is not None:
            points = self._outer_wall_contour.points
            x_pixel_coord = points[:, 0] / pixel_dims[0] + image_shape[1] / 2
            y_pixel_coord = points[:, 1] / pixel_dims[1] + image_shape[1] / 2
            rr, cc = polygon(y_pixel_coord, x_pixel_coord, image_shape)
            mask[rr, cc] = self._dataset_config.wall_value

        points = self._lumen_contour.points
        rr, cc = polygon(points[:, 1] / pixel_dims[1] + image_shape[1] / 2, points[:, 0] / pixel_dims[0] + image_shape[1] / 2, image_shape)
        mask[rr, cc] = self._dataset_config.lumen_value

        return mask

    @property
    def is_ending_cross_section(self):
        return self._ending_normal is not None

    @property
    def ending_normal(self):
        if not self.is_ending_cross_section:
            raise RuntimeError("This cross section is not an ending cross section")

        normal_direction = np.dot(self.plane_normal[:, 0], self._ending_normal[:, 0]) > 0
        if normal_direction:
            return self.plane_normal
        return -self.plane_normal


class ContourDoesNotExistError(AttributeError):
    pass
