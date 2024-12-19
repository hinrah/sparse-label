import unittest
from unittest.mock import MagicMock
from sparselabel.dataset_characteristics_extraction import get_max_voxel_size, get_min_lumen_centerline_distance, get_max_contour_centerline_distance, \
    get_median_voxel_size


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

    def test_get_min_lumen_centerline_distance(self):
        case1 = MagicMock()
        case1.min_lumen_centerline_distance.return_value = 1.0
        case2 = MagicMock()
        case2.min_lumen_centerline_distance.return_value = 2.0
        cases = [case1]*10 + [case2]*100
        self.assertEqual(1.0, get_min_lumen_centerline_distance(cases))

    def test_get_min_lumen_centerline_distance_very_few_cases(self):
        case1 = MagicMock()
        case1.min_lumen_centerline_distance.return_value = 1.0
        case2 = MagicMock()
        case2.min_lumen_centerline_distance.return_value = 2.0
        cases = [case1]*2 + [case2]*100
        self.assertEqual(2.0, get_min_lumen_centerline_distance(cases))

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
