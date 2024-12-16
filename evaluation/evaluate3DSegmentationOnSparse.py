import argparse
from multiprocessing import Lock, Manager
from multiprocessing.pool import Pool

from tqdm import tqdm

from case import EvaluationCase, CaseLoader
from evaluation.segmentation_evaluator import SegmentationEvaluator2DContourOn3DLabel, SegmentationEvaluator2DContourOn2DCrossSections
from evaluation.segmentation_results import SegmentationResults


class EvaluationProcessor:
    lock = Lock()

    def __init__(self, case_loader, manager, evaluator, with_wall=True):
        self.segmentation_results = manager.list()
        self._case_loader = case_loader
        self._evaluator = evaluator
        self._with_wall = with_wall

    def _is_valid_evaluation_cross_section(self, cross_section):
        if cross_section.lumen_points is None:
            return False
        if self._with_wall and cross_section.outer_wall_points is None:
            return False
        return True
        

    def _process_one_item_parallel(self, case):
        for cross_section in case.cross_sections:
            if not self._is_valid_evaluation_cross_section(cross_section):
                continue
            segmentation_result = self._evaluator.evaluate(cross_section, case)
            with self.lock:
                self.segmentation_results.append(segmentation_result)

    def process_parallel(self, num_threads=4):
        with Pool(processes=num_threads) as pool, tqdm(total=len(self._case_loader)) as pbar:
            for _ in pool.imap(self._process_one_item_parallel, self._case_loader):
                pbar.update()
                pbar.refresh()


def evaluate_segmentations(dataset, trainer, configuration, num_threads, evaluator, postprocessed, with_wall=True):

    manager = Manager()

    case_loader = CaseLoader(dataset, EvaluationCase, trainer=trainer, configuration=configuration, postprocessed=postprocessed)

    processor = EvaluationProcessor(case_loader, manager, evaluator, with_wall=with_wall)
    processor.process_parallel(num_threads=num_threads)

    segmentation_results = SegmentationResults()
    for segmentation_result in processor.segmentation_results:
        segmentation_results.add(segmentation_result)

    return segmentation_results
