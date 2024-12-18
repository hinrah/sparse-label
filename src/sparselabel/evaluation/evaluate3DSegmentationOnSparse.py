from multiprocessing import Lock, Manager
from multiprocessing.pool import Pool

from tqdm import tqdm

from sparselabel.data_handlers.case import EvaluationCase
from sparselabel.case_loader import CaseLoader
from sparselabel.evaluation.segmentation_results import SegmentationResults


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

    def process_synchronous(self):
        for case in tqdm(self._case_loader):
            self._process_one_item_parallel(case)


def evaluate_segmentations(dataset_config, num_threads, evaluator):
    manager = Manager()

    case_loader = CaseLoader(dataset_config, EvaluationCase)

    processor = EvaluationProcessor(case_loader, manager, evaluator, with_wall=dataset_config.has_wall)

    if num_threads <= 1:
        processor.process_synchronous()
    else:
        processor.process_parallel(num_threads=num_threads)

    segmentation_results = SegmentationResults()
    for segmentation_result in processor.segmentation_results:
        segmentation_results.add(segmentation_result)

    return segmentation_results
