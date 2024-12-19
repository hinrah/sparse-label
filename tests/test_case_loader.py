import unittest
from unittest.mock import patch, MagicMock
from sparselabel.case_loader import CaseLoader
from sparselabel.constants import Endings


class CaseMock(MagicMock):
    def __init__(self, case_id, dataset_config, **kwargs):
        super().__init__()
        self.case_id = case_id
        self.dataset_config = dataset_config
        self.kwargs = kwargs
        self.load = MagicMock()


class TestCaseLoader(unittest.TestCase):
    # pylint: disable=arguments-differ
    @patch('sparselabel.case_loader.glob')
    def setUp(self, mock_glob):
        self.dataset_config = MagicMock()
        self.dataset_config.images_path = 'mocked_images_path'
        mock_glob.return_value = [
            'mocked_images_path/case1' + Endings.CHANNEL_ZERO + Endings.NIFTI,
            'mocked_images_path/case2' + Endings.CHANNEL_ZERO + Endings.NIFTI
        ]
        self.case_loader = CaseLoader(self.dataset_config, CaseMock)

    @patch('sparselabel.case_loader.glob')
    def test_search_cases(self, mock_glob):
        mock_glob.return_value = [
            'mocked_images_path/case1' + Endings.CHANNEL_ZERO + Endings.NIFTI,
            'mocked_images_path/case2' + Endings.CHANNEL_ZERO + Endings.NIFTI
        ]
        self.case_loader._search_cases()
        self.assertEqual(['case1', 'case2'], self.case_loader.case_ids)

    def test_len(self):
        self.assertEqual(2, len(self.case_loader))

    def test_iter(self):
        iterator = iter(self.case_loader)
        self.assertEqual(iterator, self.case_loader)
        self.assertEqual(0, self.case_loader.index)

    def test_next(self):
        case = next(self.case_loader)
        self.assertEqual('case1', case.case_id)
        self.assertEqual(1, self.case_loader.index)

    def test_next_stop_iteration(self):
        next(self.case_loader)
        next(self.case_loader)
        with self.assertRaises(StopIteration):
            next(self.case_loader)

    @patch('sparselabel.case_loader.CaseLoader._search_cases')
    def test_init_calls_search_cases(self, mock_search_cases):
        CaseLoader(self.dataset_config, MagicMock)
        mock_search_cases.assert_called_once()
