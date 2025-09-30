import unittest
from unittest.mock import patch, mock_open
import os
from sparselabel.dataset_config import DatasetConfig
from sparselabel.constants import Folders, DatasetInfo


class TestDatasetConfig(unittest.TestCase):

    @patch('sparselabel.dataset_config.DatasetConfig._read_from_environment_variable_with_fallback')
    def setUp(self, mock_get_env_var):  # pylint: disable=arguments-differ
        mock_get_env_var.side_effect = ['mocked_data_raw', 'mocked_data_results']
        self.dataset_config = DatasetConfig('test_dataset')

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 10, "lumen": 11, "wall": 12}}')
    def test_dataset_info(self, _):
        info = self.dataset_config.dataset_info
        self.assertEqual(10, info['labels']['background'])
        self.assertEqual(11, info['labels']['lumen'])
        self.assertEqual(12, info['labels']['wall'])

    def test_contours_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset', Folders.CONTOURS + 'Tr')
        self.assertEqual(expected_path, self.dataset_config.contours_path)

    def test_images_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset', Folders.IMAGES + 'Tr')
        self.assertEqual(expected_path, self.dataset_config.images_path)

    def test_prediction_path(self):
        expected_path = os.path.join('mocked_data_results', 'test_dataset', 'nnUNetTrainer__nnUNetPlans__3d_fullres', 'crossval_results_folds_0_1_2_3_4')
        self.assertEqual(expected_path, self.dataset_config.prediction_path)

    def test_centerlines_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset', Folders.CENTERLINES + 'Tr')
        self.assertEqual(expected_path, self.dataset_config.centerlines_path)

    def test_labels_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset', Folders.LABELS + 'Tr')
        self.assertEqual(expected_path, self.dataset_config.labels_path)

    def test_results_path(self):
        expected_path = os.path.join('mocked_data_results', 'test_dataset')
        self.assertEqual(expected_path, self.dataset_config.results_path)

    def test_raw_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset')
        self.assertEqual(expected_path, self.dataset_config.raw_path)

    def test_dataset_info_path(self):
        expected_path = os.path.join('mocked_data_raw', 'test_dataset', DatasetInfo.FILE_NAME)
        self.assertEqual(expected_path, self.dataset_config.dataset_info_path)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "lumen": 123, "wall": 2}}')
    def test_class_value_by_label(self, _):
        self.dataset_config._dataset_info = None
        value = self.dataset_config.class_value_by_label('lumen')
        self.assertEqual(123, value)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "Lumen": 123, "Wall": 2}}')
    def test_lumen_value(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual(123, self.dataset_config.lumen_value)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "lumen": 123, "Wall": 2}}')
    def test_lumen_value_wrong_key(self, _):
        self.dataset_config._dataset_info = None
        with self.assertRaises(KeyError):
            _ = self.dataset_config.lumen_value

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 10, "lumen": 1, "wall": 2}}')
    def test_background_value(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual(10, self.dataset_config.background_value)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "lumen": 1, "Wall": 23}}')
    def test_wall_value(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual(23, self.dataset_config.wall_value)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "lumen": 1, "wall": 2, "ignore": 5}}')
    def test_wall_value_wrong_dict(self, _):
        self.dataset_config._dataset_info = None
        with self.assertRaises(KeyError):
            _ = self.dataset_config.wall_value

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "lumen": 1, "Wall": 2, "ignore": 5}}')
    def test_ignore_value(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual(5, self.dataset_config.ignore_value)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "Lumen": 1, "Wall": 2}}')
    def test_has_wall(self, _):
        self.dataset_config._dataset_info = None
        self.assertTrue(self.dataset_config.has_wall)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "Lumen": 1}}')
    def test_classes_without_wall(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual([DatasetInfo.BACKGROUND, DatasetInfo.LUMEN], self.dataset_config.classes)

    @patch('sparselabel.dataset_config.open', new_callable=mock_open, read_data='{"labels": {"background": 0, "Lumen": 1, "Wall": 2}}')
    def test_classes_with_wall(self, _):
        self.dataset_config._dataset_info = None
        self.assertEqual([DatasetInfo.BACKGROUND, DatasetInfo.WALL, DatasetInfo.LUMEN], self.dataset_config.classes)
