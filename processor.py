from copy import deepcopy
from multiprocessing import Pool

from tqdm import tqdm


class Processor:
    def __init__(self, case_handler, case_loader):
        self._case_loader = case_loader
        self._case_handler = case_handler

    def _process_one_item_parallel(self, case):
        case_handler = deepcopy(self._case_handler)
        case_handler.apply(case)

    def process(self):
        for case in self._case_loader:
            self._case_handler.apply(case)

    def process_parallel(self, num_threads=4):
        with Pool(processes=num_threads) as pool, tqdm(total=len(self._case_loader)) as pbar:
            for _ in pool.imap(self._process_one_item_parallel, self._case_loader):
                pbar.update()
                pbar.refresh()
