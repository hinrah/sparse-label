import numpy as np
from shapely import Polygon, Point
from shapely.geometry.base import BaseGeometry


class Contour:
    def __init__(self, points: np.ndarray) -> None:
        self._points = points
        self._polygon = Polygon(points)

    def contains_point(self, point: np.ndarray) -> bool:
        return self._polygon.contains(Point(point))

    def contains(self, arbitrary_shape: BaseGeometry) -> bool:
        return self._polygon.contains(arbitrary_shape)

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def points(self) -> np.ndarray:
        return self._points
