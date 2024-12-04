import shutil
from unittest import TestCase
import os
import nibabel as nib
import numpy as np
from unittest.mock import patch
from case import Case
from constants import Labels
from label_creator import LabelCreator
from labeling_strategies import LabelCrossSections, LabelCenterline


class TestDefaultLabelCreator(TestCase):

    def setUp(self) -> None:
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    @patch("case.data_raw", new=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data"))
    @patch("label_creator.data_raw", new=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data"))
    def test_create_label_with_wall(self):
        all_with_wall = [LabelCrossSections(0.6/2, with_wall=True, radius=20), LabelCenterline(0.6, Labels.LUMEN), LabelCenterline(15, Labels.BACKGROUND)]
        all_without_wall = [LabelCrossSections(0.6/2, with_wall=False, radius=20), LabelCenterline(0.6, Labels.LUMEN), LabelCenterline(15, Labels.BACKGROUND)]

        all_with_wall_label = nib.load(os.path.join(self.test_dir, "test_data", "Dataset001_test", "expected_labels", "test_with_wall.nii.gz"))
        all_without_wall_label = nib.load(os.path.join(self.test_dir, "test_data", "Dataset001_test", "expected_labels", "test_without_wall.nii.gz"))

        tests = [["all with wall", all_with_wall, all_with_wall_label],
                 ["all without wall", all_without_wall, all_without_wall_label],
                 ["all commutative", all_with_wall[::-1], all_with_wall_label]]

        for name, strategies, expected_result in tests:
            with self.subTest(name):

                label_creator = LabelCreator(strategies)

                test_case = Case("test", "Dataset001_test")
                test_case.load()

                label_creator.apply(test_case)

                true_labels = nib.load(os.path.join(self.test_dir, "test_data", "Dataset001_test", "labelsTr", "test.nii.gz"))
                np.testing.assert_array_equal(true_labels.get_fdata(), expected_result.get_fdata())
                np.testing.assert_array_equal(true_labels.affine, expected_result.affine)

    def tearDown(self) -> None:
        pass#shutil.rmtree(os.path.join(self.test_dir, "test_data", "Dataset001_test", "labelsTr"))