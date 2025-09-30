import numpy as np
from sklearn.decomposition import PCA
from skimage.draw import polygon  # pylint: disable=no-name-in-module

from sparselabel.data_handlers.contour import Contour
from sparselabel.dataset_config import DatasetConfig


class CrossSection:
    def __init__(self, dataset_config: DatasetConfig, identifier: str, inner_contour: np.ndarray, outer_contour=None, ending_normal=None) -> None:
        self._dataset_config = dataset_config
        self.identifier = identifier
        self._inner_contour_points = inner_contour
        self._outer_wall_contour_points = outer_contour
        self._pca = self._create_pca(inner_contour, outer_contour)
        self._ending_normal = ending_normal

        self._inner_contour = Contour(self.transform_points_to_plane_coordinates(inner_contour))
        if outer_contour is not None:
            self._outer_wall_contour = Contour(self.transform_points_to_plane_coordinates(outer_contour))
        else:
            self._outer_wall_contour = None

    def lumen_is_inside_wall(self) -> bool:
        if self._outer_wall_contour is None:
            raise ContourDoesNotExistError
        return self._outer_wall_contour.contains(self._inner_contour.polygon)

    @property
    def all_contour_points(self) -> np.ndarray:
        if self._outer_wall_contour_points is None:
            return self._inner_contour_points
        return np.vstack((self._inner_contour_points, self._outer_wall_contour_points))

    @property
    def inner_contour_points(self) -> np.ndarray:
        return self._inner_contour_points

    @property
    def outer_wall_points(self) -> np.ndarray:
        return self._outer_wall_contour_points

    def _create_pca(self, inner_contour_points: np.ndarray, outer_wall_contour_points: np.ndarray) -> PCA:
        if outer_wall_contour_points is None:
            points = inner_contour_points
        else:
            points = np.vstack((inner_contour_points, outer_wall_contour_points))
        pca = PCA(n_components=3, svd_solver="full")
        pca.fit(points)
        return pca

    @property
    def plane_normal(self) -> np.ndarray:
        return self._pca.components_[2:3].T

    @property
    def plane_center(self) -> np.ndarray:
        return self._pca.mean_

    @property
    def plane_transform(self) -> np.ndarray:
        return self._pca.components_[:2].T

    @property
    def plane_x_axis(self) -> np.ndarray:
        return self._pca.components_[0]

    @property
    def plane_y_axis(self) -> np.ndarray:
        return self._pca.components_[1]

    def transform_points_to_plane_coordinates(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 3)
        return (points - self.plane_center) @ self.plane_transform

    def distance_to_plane(self, points: np.ndarray) -> np.ndarray:
        return np.abs((points - self.plane_center) @ self.plane_normal)

    def is_projected_inside_lumen(self, point: np.ndarray) -> bool:
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._inner_contour.contains_point(projected_point)

    def projected_inside_lumen(self, points: np.ndarray) -> np.ndarray:
        return np.array([self.is_projected_inside_lumen(point) for point in points]).reshape(-1, 1)

    def projected_inside_wall(self, points: np.ndarray) -> np.ndarray:
        return np.array([self.is_projected_inside_wall(point) for point in points]).reshape(-1, 1)

    def is_projected_inside_wall(self, point: np.ndarray) -> bool:
        if self._outer_wall_contour is None:
            raise ContourDoesNotExistError
        projected_point = self.transform_points_to_plane_coordinates(point)[0]
        return self._outer_wall_contour.contains_point(projected_point)

    def create_pixel_mask(self, pixel_dims: tuple[float, float], image_shape: tuple[int, int]) -> np.ndarray:
        mask = np.ones(image_shape, dtype=np.uint8) * self._dataset_config.background_value

        if self._outer_wall_contour is not None:
            points = self._outer_wall_contour.points
            x_pixel_coord = points[:, 0] / pixel_dims[0] + image_shape[1] / 2
            y_pixel_coord = points[:, 1] / pixel_dims[1] + image_shape[1] / 2
            rr, cc = polygon(y_pixel_coord, x_pixel_coord, image_shape)
            mask[rr, cc] = self._dataset_config.wall_value

        points = self._inner_contour.points
        rr, cc = polygon(points[:, 1] / pixel_dims[1] + image_shape[1] / 2, points[:, 0] / pixel_dims[0] + image_shape[1] / 2, image_shape)
        mask[rr, cc] = self._dataset_config.lumen_value

        return mask

    @property
    def is_ending_cross_section(self) -> bool:
        return self._ending_normal is not None

    @property
    def ending_normal(self) -> np.ndarray:
        if not self.is_ending_cross_section:
            raise RuntimeError("This cross section is not an ending cross section")

        normal_direction = np.dot(self.plane_normal[:, 0], self._ending_normal[:, 0]) > 0
        if normal_direction:
            return self.plane_normal
        return -self.plane_normal


class ContourDoesNotExistError(AttributeError):
    pass
