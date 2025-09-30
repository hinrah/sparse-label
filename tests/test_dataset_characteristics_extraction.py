import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from sparselabel.dataset_characteristics_extraction import get_max_voxel_size, get_min_lumen_centerline_distance, get_max_contour_centerline_distance, \
    get_median_voxel_size, min_lumen_centerline_distance_one_case


class TestDatasetCharacteristicsExtraction(unittest.TestCase):

    def test_get_max_voxel_size(self):
        case1 = MagicMock()
        case1.voxel_size = [1, 2, 3]
        case2 = MagicMock()
        case2.voxel_size = [2, 3, 4]
        cases = [case1, case2]
        self.assertEqual(4, get_max_voxel_size(cases))

    def test_get_max_voxel_size_no_cases(self):
        with self.assertRaises(ValueError):
            get_max_voxel_size([])

    def test_get_median_voxel_size(self):
        case1 = MagicMock()
        case1.voxel_size = [1, 2, 3]
        case2 = MagicMock()
        case2.voxel_size = [3, 3, 4]
        cases = [case1, case2]
        self.assertEqual(3, get_median_voxel_size(cases))

    def test_get_median_voxel_size_no_cases(self):
        with self.assertRaises(ValueError):
            get_median_voxel_size([])

    @patch('sparselabel.dataset_characteristics_extraction.min_lumen_centerline_distance_one_case')
    def test_get_min_lumen_centerline_distance(self, mock_min_lumen_centerline_distance_one_case):
        mock_min_lumen_centerline_distance_one_case.side_effect = [1.0] * 10 + [2.0] * 100

        self.assertEqual(1.0, get_min_lumen_centerline_distance([MagicMock()] * 110))

    @patch('sparselabel.dataset_characteristics_extraction.min_lumen_centerline_distance_one_case')
    def test_get_min_lumen_centerline_distance_very_few_cases(self, mock_min_lumen_centerline_distance_one_case):
        mock_min_lumen_centerline_distance_one_case.side_effect = [1.0] * 2 + [2.0] * 100

        self.assertEqual(2.0, get_min_lumen_centerline_distance([MagicMock()] * 102))

    def test_min_lumen_centerline_distance_zero(self):
        case = MagicMock()
        case.all_centerline_points.return_value = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        case.all_inner_contour_points.return_value = np.array([[1, 1, 1], [3, 3, 3]])
        result = min_lumen_centerline_distance_one_case(case)
        self.assertEqual(result, 0.0)

    def test_min_lumen_centerline_distance(self):
        case = MagicMock()
        case.all_centerline_points.return_value = np.array([[0, 0, 1], [2, 2, 1], [3, 2, 2]])
        case.all_inner_contour_points.return_value = np.array([[0, 0, 0], [7, 7, 7]])

        result = min_lumen_centerline_distance_one_case(case)
        self.assertEqual(result, 1.0)

    def test_min_lumen_centerline_distance_with_empty_points(self):
        test_cases = [
            (np.array([]), np.array([[1, 1, 1], [3, 3, 3]])),
            (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), np.array([]))
        ]

        for centerline_points, inner_contour_points in test_cases:
            with self.subTest(centerline_points=centerline_points, lumen_points=inner_contour_points):
                case = MagicMock()
                case.all_centerline_points.return_value = centerline_points
                case.all_inner_contour_points.return_value = inner_contour_points

                with self.assertRaises(ValueError):
                    min_lumen_centerline_distance_one_case(case)

    def test_get_min_lumen_centerline_distance_no_valid_cases(self):
        case1 = MagicMock()
        case1.min_lumen_centerline_distance.side_effect = ValueError
        cases = [case1]
        with self.assertRaises(IndexError):
            get_min_lumen_centerline_distance(cases)

    def test_get_max_contour_centerline_distance(self):
        case1 = MagicMock()
        case1.max_contour_centerline_distance.return_value = 1.0
        case2 = MagicMock()
        case2.max_contour_centerline_distance.return_value = 2.0
        cases = [case1, case2]
        self.assertEqual(2.0, get_max_contour_centerline_distance(cases))

    def test_get_max_contour_centerline_distance_no_valid_cases(self):
        case1 = MagicMock()
        case1.max_contour_centerline_distance.side_effect = ValueError
        cases = [case1]
        with self.assertRaises(ValueError):
            get_max_contour_centerline_distance(cases)
