from unittest import TestCase
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

from sparselabel.data_handlers.evaluation_case import EvaluationCase


class TestClassCase(TestCase):

    @patch('sparselabel.data_handlers.evaluation_case.EvaluationCase.cross_sections', new_callable=PropertyMock)
    def test_true_outer_wall_points_with_valid_data(self, mock_cross_sections):
        cross_section_1 = MagicMock(outer_wall_points=np.array([[0, 0, 0], [1, 1, 1]]))
        cross_section_2 = MagicMock(outer_wall_points=np.array([[2, 2, 2]]))
        evaluation_case = EvaluationCase('test_case', MagicMock())
        mock_cross_sections.return_value = [cross_section_1, cross_section_2]

        result = evaluation_case.true_outer_wall_points()

        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_equal(result, expected)

    @patch('sparselabel.data_handlers.evaluation_case.EvaluationCase.cross_sections', new_callable=PropertyMock)
    def test_true_outer_wall_points_with_no_valid_data(self, mock_cross_sections):
        evaluation_case = EvaluationCase('test_case', MagicMock())
        mock_cross_sections.return_value = []

        result = evaluation_case.true_outer_wall_points()

        self.assertIsNone(result)

    @patch('sparselabel.data_handlers.evaluation_case.EvaluationCase.cross_sections', new_callable=PropertyMock)
    def test_true_lumen_points_with_valid_data(self, mock_cross_sections):
        cross_section_1 = MagicMock(inner_contour_points=np.array([[0, 0, 0], [1, 1, 1]]))
        cross_section_2 = MagicMock(inner_contour_points=np.array([[2, 2, 2]]))
        evaluation_case = EvaluationCase('test_case', MagicMock())
        mock_cross_sections.return_value = [cross_section_1, cross_section_2]

        result = evaluation_case.true_lumen_points()

        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_equal(result, expected)

    @patch('sparselabel.data_handlers.evaluation_case.EvaluationCase.cross_sections', new_callable=PropertyMock)
    def test_true_lumen_points_with_no_valid_data(self, mock_cross_sections):
        evaluation_case = EvaluationCase('test_case', MagicMock())
        mock_cross_sections.return_value = []

        result = evaluation_case.true_lumen_points()

        self.assertIsNone(result)
