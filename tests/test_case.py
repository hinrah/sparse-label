import os
from unittest import TestCase
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
from networkx import DiGraph
from networkx.readwrite import json_graph

from case import Case
from constants import Contours, Folders, data_raw, Endings


class TestCase(TestCase):
    @patch.object(Case, '_load_image')
    @patch.object(Case, '_load_cross_sections')
    @patch.object(Case, '_load_centerline')
    def test_Case_construction_does_not_load(self, mock_load_centerline, mock_load_cross_sections, mock_load_image):
        case = Case('test_case', 'test_dataset')
        mock_load_image.assert_not_called()
        mock_load_cross_sections.assert_not_called()
        mock_load_centerline.assert_not_called()

    @patch.object(Case, '_load_image')
    @patch.object(Case, '_load_cross_sections')
    @patch.object(Case, '_load_centerline')
    def test_load(self, mock_load_centerline, mock_load_cross_sections, mock_load_image):
        case = Case('test_case', 'test_dataset')
        case.load()
        mock_load_image.assert_called_once()
        mock_load_cross_sections.assert_called_once()
        mock_load_centerline.assert_called_once()

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_with_valid_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                Contours.OUTER: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
            },
            "section2": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                Contours.OUTER: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
            }
        }
        case = Case('test_case', 'test_dataset')
        case._load_cross_sections()
        self.assertEqual(len(case.cross_sections), 2)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_with_missing_contours(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1]]
            },
            "section2": {
                Contours.OUTER: [[6, 6, 6], [7, 7, 7]]
            }
        }
        case = Case('test_case', 'test_dataset')
        case._load_cross_sections()
        self.assertEqual(len(case.cross_sections), 0)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_with_empty_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {}
        case = Case('test_case', 'test_dataset')
        case._load_cross_sections()
        self.assertEqual(len(case.cross_sections), 0)

    @patch('builtins.open', new_callable=mock_open, read_data='{"inner_contour": [[0, 0, 0]], "outer_contour": [[1, 1, 1]]}')
    @patch('case.os.path.join', return_value='mocked_path')
    def test_load_raw_cross_sections_with_valid_data(self, mock_join, mock_open):
        case = Case('test_case', 'test_dataset')
        result = case._load_raw_cross_sections()
        expected = {
            "inner_contour": [[0, 0, 0]],
            "outer_contour": [[1, 1, 1]]
        }
        self.assertEqual(result, expected)

    @patch('case.nib.load')
    @patch('case.os.path.join')
    def test__load_image(self, mock_join, mock_nib_load):
        case = Case('test_case', 'test_dataset')
        case._load_image()

        mock_join.assert_called_once_with(data_raw, 'test_dataset', Folders.IMAGES, 'test_case' + Endings.CHANNEL_ZERO + Endings.NIFTI)
        mock_nib_load.assert_called_once_with(mock_join.return_value)
        self.assertEqual(case.image, mock_nib_load.return_value)

    @patch('builtins.open', new_callable=mock_open, read_data='{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"id": 1}, {"id": 2}], "edges": [{"source": 1, "target": 2}]}')
    @patch('case.os.path.join', return_value='mocked_path')
    def test__load_centerline(self, mock_join, mock_open):
        case = Case('test_case', 'test_dataset')
        case._load_centerline()

        mock_join.assert_called_once_with(data_raw, 'test_dataset', Folders.CENTERLINES, 'test_case' + Endings.JSON)
        mock_open.assert_called_once_with('mocked_path', 'r')
        self.assertIsInstance(case.centerline, DiGraph)
        self.assertEqual(len(case.centerline.nodes), 2)
        self.assertEqual(len(case.centerline.edges), 1)

    def test_image_shape(self):
        case = Case('test_case', 'test_dataset')
        case.image = MagicMock()

        self.assertEqual(case.image_shape, case.image.shape)

    def test_affine(self):
        case = Case('test_case', 'test_dataset')
        case.image = MagicMock()

        self.assertEqual(case.affine, case.image.affine)

    def test_voxel_size(self):
        case = Case('test_case', 'test_dataset')
        case.image = MagicMock(header = {'pixdim': [0, 1, 2, 3, 4, 5, 6]})

        self.assertEqual(case.voxel_size, [1,2,3])
    @patch.object(Case, '_all_centerline_points')
    @patch.object(Case, '_all_lumen_points')
    def test_min_lumen_centerline_distance_zero(self, mock_all_lumen_points, mock_all_centerline_points):
        mock_all_centerline_points.return_value = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        mock_all_lumen_points.return_value = np.array([[1, 1, 1], [3, 3, 3]])

        case = Case('test_case', 'test_dataset')
        result = case.min_lumen_centerline_distance()
        self.assertEqual(result, 0.0)

    @patch.object(Case, '_all_centerline_points')
    @patch.object(Case, '_all_lumen_points')
    def test_min_lumen_centerline_distance_zero(self, mock_all_lumen_points, mock_all_centerline_points):
        mock_all_centerline_points.return_value = np.array([[0, 0, 1], [2, 2, 1], [3, 2, 2]])
        mock_all_lumen_points.return_value = np.array([[0, 0, 0], [7, 7, 7]])

        case = Case('test_case', 'test_dataset')
        result = case.min_lumen_centerline_distance()
        self.assertEqual(result, 1.0)

    @patch.object(Case, '_all_centerline_points')
    @patch.object(Case, '_all_lumen_points')
    def test_min_lumen_centerline_distance_with_empty_points(self, mock_all_lumen_points, mock_all_centerline_points):
        test_cases = [
            (np.array([]), np.array([[1, 1, 1], [3, 3, 3]])),
            (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), np.array([]))
        ]

        for centerline_points, lumen_points in test_cases:
            with self.subTest(centerline_points=centerline_points, lumen_points=lumen_points):
                mock_all_centerline_points.return_value = centerline_points
                mock_all_lumen_points.return_value = lumen_points

                case = Case('test_case', 'test_dataset')
                with self.assertRaises(ValueError):
                    case.min_lumen_centerline_distance()

    @patch.object(Case, '_all_contour_points')
    @patch.object(Case, '_load_raw_cross_sections')
    def test_max_contour_centerline_distance(self, mock_load_raw_cross_sections, mock_all_contour_points):
        mock_all_contour_points.return_value = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        case = Case('test_case', 'test_dataset')
        case.cross_sections = [MagicMock(plane_center=np.array([0, 0, 0])),
                               MagicMock(plane_center=np.array([1, 1, 1])),
                               MagicMock(plane_center=np.array([2, 2, 2]))]

        result = case.max_contour_centerline_distance()
        self.assertEqual(result, 0.0)

    @patch.object(Case, '_all_contour_points')
    @patch.object(Case, '_load_raw_cross_sections')
    def test_max_contour_centerline_distance_with_empty_points(self, mock_load_raw_cross_sections, mock_all_contour_points):
        test_cases = [
            (np.array([]), np.array([[1, 1, 1], [3, 3, 3]])),
            ([MagicMock(plane_center=np.array([0, 0, 0]))], np.array([]))
        ]

        for cross_sections, contour_points in test_cases:
            with self.subTest(cross_sections=cross_sections, contour_points=contour_points):
                mock_all_contour_points.return_value = contour_points
                case = Case('test_case', 'test_dataset')
                case.cross_sections = cross_sections
                with self.assertRaises(ValueError):
                    case.max_contour_centerline_distance()

    @patch.object(Case, '_load_raw_cross_sections')
    def test_all_contour_points_with_valid_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1]],
                Contours.OUTER: [[2, 2, 2], [3, 3, 3]]
            },
            "section2": {
                Contours.INNER: [[4, 4, 4], [5, 5, 5]],
                Contours.OUTER: [[6, 6, 6], [7, 7, 7]]
            }
        }
        case = Case('test_case', 'test_dataset')
        result = case._all_contour_points()
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
        np.testing.assert_array_equal(result, expected)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_all_contour_points_with_missing_contours(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1]]
            },
            "section2": {
                Contours.OUTER: [[6, 6, 6], [7, 7, 7]]
            }
        }
        case = Case('test_case', 'test_dataset')
        result = case._all_contour_points()
        expected = np.zeros((0, 3))
        np.testing.assert_array_equal(result, expected)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_all_contour_points_with_empty_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {}
        case = Case('test_case', 'test_dataset')
        result = case._all_contour_points()
        expected = np.zeros((0, 3))
        np.testing.assert_array_equal(result, expected)
    @patch.object(Case, '_load_raw_cross_sections')
    def test_all_lumen_points_with_valid_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1]]
            },
            "section2": {
                Contours.INNER: [[2, 2, 2], [3, 3, 3]]
            }
        }
        case = Case('test_case', 'test_dataset')
        result = case._all_lumen_points()
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(result, expected)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_all_lumen_points_with_missing_inner_contour(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.OUTER: [[0, 0, 0], [1, 1, 1]]
            },
            "section2": {
                Contours.INNER: [[2, 2, 2], [3, 3, 3]]
            }
        }
        case = Case('test_case', 'test_dataset')
        result = case._all_lumen_points()
        expected = np.array([[2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_all_centerline_points_with_valid_data(self):
        case = Case('test_case', 'test_dataset')
        centerline_raw = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": 1},
                      {"id": 2},
                      {"id": 3}],
            "edges": [{"source": 1, "target": 2, 'skeletons': np.array([[0, 0, 0], [1, 1, 1]])},
                      {"source": 2, "target": 3, 'skeletons': np.array([[2, 2, 2], [3, 3, 3]])}]
        }
        case.centerline = json_graph.node_link_graph(centerline_raw, link="edges")

        result = case._all_centerline_points()
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(result, expected)

    @patch('case.data_raw')
    @patch('case.nib.save')
    @patch('case.nib.Nifti1Image')
    def test_save_label(self, mock_nifti1image, mock_save, mock_data_raw):
        case = Case('test_case', 'test_dataset')
        case.image = MagicMock()
        voxel_mask = np.array([[0, 1], [2, 3]])

        case.save_label(voxel_mask)

        np.testing.assert_array_equal(mock_nifti1image.call_args[0][0], np.astype(voxel_mask, np.int16))
        self.assertEqual(mock_nifti1image.call_args[0][1], case.image.affine)
        mock_save.assert_called_once_with(mock_nifti1image.return_value, os.path.join(mock_data_raw, 'test_dataset', Folders.LABELS, 'test_case' + Endings.NIFTI))
