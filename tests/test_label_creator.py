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
    def test_create_label(self):
        strategies = [LabelCrossSections(0.6/2), LabelCenterline(0.6, Labels.LUMEN), LabelCenterline(15, Labels.BACKGROUND)]
        label_creator = LabelCreator(strategies)

        test_case = Case("test", "Dataset001_test")
        test_case.load()

        label_creator.create_label(test_case)

        expected_labels = nib.load(os.path.join(self.test_dir, "test_data", "Dataset001_test", "expected_labels", "test.nii.gz"))
        true_labels = nib.load(os.path.join(self.test_dir, "test_data", "Dataset001_test", "labels", "test.nii.gz"))
        np.testing.assert_array_equal(true_labels.get_fdata(), expected_labels.get_fdata())
        np.testing.assert_array_equal(true_labels.affine, expected_labels.affine)

    def tearDown(self) -> None:
        pass#shutil.rmtree(os.path.join(self.test_dir, "test_data", "Dataset001_test", "labels"))