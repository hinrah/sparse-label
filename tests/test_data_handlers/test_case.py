from unittest import TestCase
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
from networkx import DiGraph
from networkx.readwrite import json_graph

from sparselabel.data_handlers.case import Case
from sparselabel.constants import Contours, Endings, ENCODING


class TestClassCase(TestCase):
    @patch.object(Case, '_load_image_properties')
    @patch.object(Case, '_load_cross_sections')
    @patch.object(Case, '_load_centerline')
    def test_Case_construction_does_not_load(self, mock_load_centerline, mock_load_cross_sections, mock_load_image_properties):
        _ = Case('test_case', 'test_dataset')
        mock_load_image_properties.assert_not_called()
        mock_load_cross_sections.assert_not_called()
        mock_load_centerline.assert_not_called()

    @patch.object(Case, '_load_image')
    @patch.object(Case, '_load_image_properties')
    @patch.object(Case, '_load_cross_sections')
    @patch.object(Case, '_load_centerline')
    def test_load(self, mock_load_centerline, mock_load_cross_sections, mock_load_image_properties, mock_load_image):
        case = Case('test_case', 'test_dataset')
        case.load()
        mock_load_image.assert_called_once()
        mock_load_image_properties.assert_called_once()
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
                Contours.INNER: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
            },
            "section2": {
                Contours.OUTER: [[6, 6, 6], [7, 7, 7], [8, 8, 8]]
            }
        }

        case = Case('test_case', 'test_dataset')
        case._load_cross_sections()

        self.assertEqual(1, len(case.cross_sections))
        np.testing.assert_array_equal(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), case.cross_sections[0].all_contour_points)

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_raises_for_to_few_points(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0, 0], [1, 1, 1]]
            }
        }
        case = Case('test_case', 'test_dataset')
        with self.assertRaises(ValueError):
            case._load_cross_sections()

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_raises_for_non_3D_points(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {
            "section1": {
                Contours.INNER: [[0, 0], [1, 1], [1, 1]]
            }
        }
        case = Case('test_case', 'test_dataset')
        with self.assertRaises(ValueError):
            case._load_cross_sections()

    @patch.object(Case, '_load_raw_cross_sections')
    def test_load_cross_sections_with_empty_data(self, mock_load_raw_cross_sections):
        mock_load_raw_cross_sections.return_value = {}
        case = Case('test_case', 'test_dataset')
        case._load_cross_sections()
        self.assertEqual(len(case.cross_sections), 0)

    @patch('builtins.open', new_callable=mock_open, read_data='{"inner_contour": [[0, 0, 0]], "outer_contour": [[1, 1, 1]]}')
    @patch('sparselabel.data_handlers.case.os.path.join', return_value='mocked_path')
    def test_load_raw_cross_sections_with_valid_data(self, *_):
        dataset_config_mock = MagicMock()
        case = Case('test_case', dataset_config_mock)
        result = case._load_raw_cross_sections()
        expected = {
            "inner_contour": [[0, 0, 0]],
            "outer_contour": [[1, 1, 1]]
        }
        self.assertEqual(result, expected)

    @patch('sparselabel.data_handlers.case.nib.load')
    @patch('sparselabel.data_handlers.case.os.path.join')
    def test__load_image_properties(self, mock_join, mock_nib_load):
        dataset_config_mock = MagicMock()
        case = Case('test_case', dataset_config_mock)
        case._load_image_properties()

        mock_join.assert_called_once_with(dataset_config_mock.images_path, 'test_case' + Endings.CHANNEL_ZERO + Endings.NIFTI)
        mock_nib_load.assert_called_once_with(mock_join.return_value)
        self.assertEqual(case.affine, mock_nib_load.return_value.affine)
        self.assertEqual(case.image_shape, mock_nib_load.return_value.shape)
        self.assertEqual(case.voxel_size, mock_nib_load.return_value.header['pixdim'][1:4])

    @patch('builtins.open', new_callable=mock_open,
           read_data='{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"id": 1}, {"id": 2}], "edges": [{"source": 1, "target": 2}]}')
    @patch('sparselabel.data_handlers.case.os.path.join', return_value='mocked_path')
    def test__load_centerline(self, mock_join, mock_open_):
        dataset_config_mock = MagicMock()
        case = Case('test_case', dataset_config_mock)
        case._load_centerline()

        mock_join.assert_called_once_with(dataset_config_mock.centerlines_path, 'test_case' + Endings.JSON)
        mock_open_.assert_called_once_with('mocked_path', 'r', encoding=ENCODING)
        self.assertIsInstance(case.centerline, DiGraph)
        self.assertEqual(len(case.centerline.nodes), 2)
        self.assertEqual(len(case.centerline.edges), 1)

    @patch.object(Case, '_all_contour_points')
    @patch.object(Case, '_load_raw_cross_sections')
    def test_max_contour_centerline_distance(self, _, mock_all_contour_points):
        mock_all_contour_points.return_value = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        case = Case('test_case', 'test_dataset')
        case.cross_sections = [MagicMock(plane_center=np.array([0, 0, 0])),
                               MagicMock(plane_center=np.array([1, 1, 1])),
                               MagicMock(plane_center=np.array([2, 2, 2]))]

        result = case.max_contour_centerline_distance()
        self.assertEqual(result, 0.0)

    @patch.object(Case, '_all_contour_points')
    @patch.object(Case, '_load_raw_cross_sections')
    def test_max_contour_centerline_distance_with_empty_points(self, _, mock_all_contour_points):
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

    def test_all_contour_points_with_valid_data(self):
        cross_section_1 = MagicMock(all_contour_points=np.array([[0, 0, 0], [1, 1, 1]]))
        cross_section_2 = MagicMock(all_contour_points=np.array([[2, 2, 2]]))
        case = Case('test_case', 'test_dataset')
        case.cross_sections = [cross_section_1, cross_section_2]

        result = case._all_contour_points()

        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_all_contour_points_with_empty_data(self):
        case = Case('test_case', 'test_dataset')
        case.cross_sections = []

        result = case._all_contour_points()

        expected = np.zeros((0, 3))
        np.testing.assert_array_equal(result, expected)

    def test_all_lumen_points_with_valid_data(self):
        cross_section_1 = MagicMock(inner_contour_points=np.array([[0, 0, 0], [1, 1, 1]]))
        cross_section_2 = MagicMock(inner_contour_points=np.array([[2, 2, 2]]))
        case = Case('test_case', 'test_dataset')
        case.cross_sections = [cross_section_1, cross_section_2]

        result = case.all_inner_contour_points()

        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
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

        result = case.all_centerline_points()
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(result, expected)
