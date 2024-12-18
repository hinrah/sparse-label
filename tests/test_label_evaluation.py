import json
from unittest import TestCase
import os
from dataset_config import DatasetConfig
from evaluation.evaluate3DSegmentationOnSparse import evaluate_segmentations
from evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn2DCrossSections, SegmentationEvaluator2DContourOn3DLabel


class TestDefaultLabelCreator(TestCase):

    def setUp(self) -> None:
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    def test_evaluate_label_2D(self):
        dataset_config = DatasetConfig("Dataset001_test",
                                       prediction_sub_path=os.path.join("test_trainer__nnUNetPlans__test_config", "crossval_results_folds_0_1_2_3_4"))
        dataset_config.data_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        dataset_config.data_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_results")

        evaluator = SegmentationEvaluator2DContourOn2DCrossSections(dataset_config, mpr_resolution=(0.0976562, 0.0976562), mpr_shape=(256, 256))
        segmentation_results = evaluate_segmentations(dataset_config, 1, evaluator)

        with open(os.path.join(dataset_config.results_path, "expected_segmentation_results", "segmentation_results.json")) as file:
            expected_segmentation_results = json.load(file)

        self.assertEqual(json.dumps(expected_segmentation_results), json.dumps(segmentation_results.to_json()))

    def test_evaluate_label_3D(self):
        dataset_config = DatasetConfig("Dataset001_test",
                                       prediction_sub_path=os.path.join("test_trainer__nnUNetPlans__test_config", "crossval_results_folds_0_1_2_3_4"))
        dataset_config.data_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        dataset_config.data_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_results")

        evaluator = SegmentationEvaluator2DContourOn3DLabel(dataset_config)
        segmentation_results = evaluate_segmentations(dataset_config, 1, evaluator)

        with open(os.path.join(dataset_config.results_path, "expected_segmentation_results", "segmentation_results_3D.json")) as file:
            expected_segmentation_results = json.load(file)

        self.assertEqual(json.dumps(expected_segmentation_results), json.dumps(segmentation_results.to_json()))

    def test_evaluate_label_3D_arbitrary_label_values(self):
        dataset_config = DatasetConfig("Dataset002_test_other_values",
                                       prediction_sub_path=os.path.join("test_trainer__nnUNetPlans__test_config", "crossval_results_folds_0_1_2_3_4"))
        dataset_config.data_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        dataset_config.data_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_results")

        evaluator = SegmentationEvaluator2DContourOn3DLabel(dataset_config)
        segmentation_results = evaluate_segmentations(dataset_config, 1, evaluator)

        with open(os.path.join(dataset_config.results_path, "expected_segmentation_results", "segmentation_results_3D.json")) as file:
            expected_segmentation_results = json.load(file)

        self.assertEqual(json.dumps(expected_segmentation_results), json.dumps(segmentation_results.to_json()))

    def test_evaluate_label_2D_arbitrary_label_values(self):
        dataset_config = DatasetConfig("Dataset002_test_other_values",
                                       prediction_sub_path=os.path.join("test_trainer__nnUNetPlans__test_config", "crossval_results_folds_0_1_2_3_4"))
        dataset_config.data_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        dataset_config.data_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_results")

        evaluator = SegmentationEvaluator2DContourOn2DCrossSections(dataset_config, mpr_resolution=(0.0976562, 0.0976562), mpr_shape=(256, 256))
        segmentation_results = evaluate_segmentations(dataset_config, 1, evaluator)

        with open(os.path.join(dataset_config.results_path, "expected_segmentation_results", "segmentation_results.json")) as file:
            expected_segmentation_results = json.load(file)

        self.assertEqual(json.dumps(expected_segmentation_results), json.dumps(segmentation_results.to_json()))
