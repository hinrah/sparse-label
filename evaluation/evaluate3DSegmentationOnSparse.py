import argparse
from multiprocessing import Lock, Manager
from multiprocessing.pool import Pool

from tqdm import tqdm

from case import EvaluationCase, CaseLoader
from evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn3DLabel, SegmentationEvaluator2DContourOn2DCrossSections
from evaluation.segmentation_results import SegmentationResults


class EvaluationProcessor:
    lock = Lock()

    def __init__(self, case_loader, manager, evaluator):
        self.segmentation_results = manager.list()
        self._case_loader = case_loader
        self._evaluator = evaluator

    def _process_one_item_parallel(self, case):
        for cross_section in case.cross_sections:
            if cross_section.lumen_points is None or cross_section.outer_wall_points is None:
                continue
            segmentation_result = self._evaluator.evaluate(cross_section, case)
            with self.lock:
                self.segmentation_results.append(segmentation_result)

    def process_parallel(self, num_threads=4):
        with Pool(processes=num_threads) as pool, tqdm(total=len(self._case_loader)) as pbar:
            for _ in pool.imap(self._process_one_item_parallel, self._case_loader):
                pbar.update()
                pbar.refresh()


def evaluate_segmentations():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="[REQUIRED] dataset name (folder name) for which the label creation is performed.")
    args = parser.parse_args()
    manager = Manager()

    evaluator = SegmentationEvaluator2DContourOn3DLabel(classes=[0, 1, 2])
    processor = EvaluationProcessor(CaseLoader(args.d, EvaluationCase), manager, evaluator)
    processor.process_parallel(num_threads=4)

    segmentation_results = SegmentationResults()
    for segmentation_result in processor.segmentation_results:
        segmentation_results.add(segmentation_result)

    return segmentation_results
