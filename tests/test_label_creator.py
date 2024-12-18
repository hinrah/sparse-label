from unittest import TestCase
import os
import shutil

import nibabel as nib
import numpy as np
from case import Case
from dataset_config import DatasetConfig
from label_creator import LabelCreator
from labeling_strategies import LabelCrossSections, LabelCenterline, LabelEndingCrossSections


class TestDefaultLabelCreator(TestCase):

    def setUp(self) -> None:
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    def test_create_label_with_wall(self):
        dataset_config = DatasetConfig("Dataset001_test")
        dataset_config.data_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

        all_with_wall = [LabelCrossSections(dataset_config, 0.6 / 2, with_wall=True, radius=20),
                         LabelCenterline(dataset_config, 0.6, dataset_config.lumen_value),
                         LabelCenterline(dataset_config, 15, dataset_config.background_value),
                         LabelEndingCrossSections(dataset_config, 0.6 / 2, 15)]
        all_without_wall = [LabelCrossSections(dataset_config, 0.6 / 2, with_wall=False, radius=20),
                            LabelCenterline(dataset_config, 0.6, dataset_config.lumen_value),
                            LabelCenterline(dataset_config, 15, dataset_config.background_value),
                            LabelEndingCrossSections(dataset_config, 0.6 / 2, 15)]

        all_with_wall_label = nib.load(os.path.join(dataset_config.raw_path, "expected_labels", "test_with_wall.nii.gz"))
        all_without_wall_label = nib.load(os.path.join(dataset_config.raw_path, "expected_labels", "test_without_wall.nii.gz"))

        tests = [["all with wall", all_with_wall, all_with_wall_label],
                 ["all without wall", all_without_wall, all_without_wall_label],
                 ["all commutative", all_with_wall[::-1], all_with_wall_label]]

        for name, strategies, expected_result in tests:
            with self.subTest(name):
                label_creator = LabelCreator(strategies, dataset_config)

                test_case = Case("test", dataset_config)
                test_case.load()

                label_creator.apply(test_case)

                true_labels = nib.load(os.path.join(dataset_config.labels_path, "test.nii.gz"))
                np.testing.assert_array_equal(true_labels.get_fdata(), expected_result.get_fdata())
                np.testing.assert_array_equal(true_labels.affine, expected_result.affine)

    def tearDown(self) -> None:
        shutil.rmtree(os.path.join(self.test_dir, "test_data", "Dataset001_test", "labelsTr"))
