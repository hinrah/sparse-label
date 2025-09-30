import unittest
import numpy as np
from unittest.mock import MagicMock
from sparselabel.data_handlers.cross_section import CrossSection, ContourDoesNotExistError


class TestCrossSection(unittest.TestCase):

    def setUp(self):
        self.dataset_config = MagicMock()
        self.dataset_config.background_value = 0
        self.dataset_config.wall_value = 1
        self.dataset_config.lumen_value = 2

        self.lumen_contour = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]])
        self.outer_wall_contour = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
        self.cross_section = CrossSection(self.dataset_config, 'test_id', self.lumen_contour, self.outer_wall_contour)

    def test_lumen_is_inside_wall(self):
        self.assertTrue(self.cross_section.inner_contour_inside_outer_contour())

    def test_lumen_is_inside_wall_no_outer_wall(self):
        cross_section = CrossSection(self.dataset_config, 'test_id', self.lumen_contour)
        with self.assertRaises(ContourDoesNotExistError):
            cross_section.inner_contour_inside_outer_contour()

    def test_all_contour_points(self):
        expected_points = np.vstack((self.lumen_contour, self.outer_wall_contour))
        np.testing.assert_array_equal(expected_points, self.cross_section.all_contour_points)

    def test_lumen_points(self):
        np.testing.assert_array_equal(self.lumen_contour, self.cross_section.inner_contour_points)

    def test_outer_wall_points(self):
        np.testing.assert_array_equal(self.outer_wall_contour, self.cross_section.outer_wall_points)

    def test_plane_normal(self):
        self.assertEqual(self.cross_section.plane_normal.shape, (3, 1))

    def test_plane_center(self):
        self.assertEqual(self.cross_section.plane_center.shape, (3,))

    def test_plane_transform(self):
        self.assertEqual(self.cross_section.plane_transform.shape, (3, 2))

    def test_transform_points_to_plane_coordinates(self):
        transformed_points = self.cross_section.transform_points_to_plane_coordinates(self.lumen_contour)
        self.assertEqual(transformed_points.shape, (4, 2))

    def test_distance_to_plane(self):
        distances = self.cross_section.distance_to_plane(self.lumen_contour)
        self.assertEqual(distances.shape, (4, 1))

    def test_is_projected_inside_lumen(self):
        point = np.array([[1, 1, 0]])
        self.assertTrue(self.cross_section.is_projected_inside_lumen(point))

    def test_projected_inside_lumen(self):
        points = np.array([[1, 1, 0], [1.5, 1.5, 0], [0.5, 0.5, 0]])
        expected = np.array([[True], [False], [False]])
        np.testing.assert_array_equal(expected, self.cross_section.projected_inside_lumen(points))

    def test_projected_inside_wall(self):
        points = np.array([[0.5, 0.5, 0], [1.5, 1.5, 0]])
        expected = np.array([[True], [True]])
        np.testing.assert_array_equal(expected, self.cross_section.projected_inside_wall(points))

    def test_is_projected_inside_wall(self):
        point = np.array([[0.5, 0.5, 0]])
        self.assertTrue(self.cross_section.is_projected_inside_wall(point))

    def test_create_pixel_mask(self):
        pixel_dims = (1, 1)
        image_shape = (10, 10)
        mask = self.cross_section.create_pixel_mask(pixel_dims, image_shape)
        self.assertEqual(mask.shape, image_shape)

    def test_is_ending_cross_section(self):
        self.assertFalse(self.cross_section.is_ending_cross_section)
        cross_section = CrossSection(self.dataset_config, 'test_id', self.lumen_contour, ending_normal=np.array([0, 0, 1]))
        self.assertTrue(cross_section.is_ending_cross_section)

    def test_ending_normal(self):
        cross_section = CrossSection(self.dataset_config, 'test_id', self.lumen_contour, ending_normal=np.array([[0, 0, 1]]).T)
        np.testing.assert_array_equal(cross_section.plane_normal, cross_section.ending_normal)
