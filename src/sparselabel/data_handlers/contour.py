from shapely import Polygon, Point


class Contour:
    def __init__(self, points):
        self._points = points
        self._polygon = Polygon(points)

    def contains_point(self, point):
        return self._polygon.contains(Point(point))

    def contains(self, arbitrary_shape):
        return self._polygon.contains(arbitrary_shape)

    @property
    def polygon(self):
        return self._polygon

    @property
    def points(self):
        return self._points
