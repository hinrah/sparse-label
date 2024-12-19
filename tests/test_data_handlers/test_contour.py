import unittest
from shapely.geometry import Polygon
from sparselabel.data_handlers.contour import Contour


class TestContour(unittest.TestCase):

    def setUp(self):
        self.points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.contour = Contour(self.points)

    def test_contains_point_inside(self):
        point = (0.5, 0.5)
        self.assertTrue(self.contour.contains_point(point))

    def test_contains_point_outside(self):
        point = (1.5, 1.5)
        self.assertFalse(self.contour.contains_point(point))

    def test_contains_shape_inside(self):
        shape = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        self.assertTrue(self.contour.contains(shape))

    def test_contains_shape_outside(self):
        shape = Polygon([(1.2, 1.2), (1.8, 1.2), (1.8, 1.8), (1.2, 1.8)])
        self.assertFalse(self.contour.contains(shape))

    def test_contains_shape_partially_outside(self):
        shape = Polygon([(0.2, 0.2), (1.8, 0.2), (1.8, 1.8), (0.2, 1.8)])
        self.assertFalse(self.contour.contains(shape))

    def test_polygon_property(self):
        self.assertIsInstance(self.contour.polygon, Polygon)
        self.assertEqual(self.contour.polygon, Polygon(self.points))

    def test_points_property(self):
        self.assertEqual(self.points, self.contour.points)
