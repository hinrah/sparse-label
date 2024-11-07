from copy import deepcopy
from multiprocessing import Pool

from tqdm import tqdm


class Processor:
    def __init__(self, label_creator, case_loader):
        self._case_loader = case_loader
        self._label_creator = label_creator

    def _process_one_item_parallel(self, case):
        label_creator = deepcopy(self._label_creator)
        label_creator.create_label(case)

    def process(self):
        for i, case in tqdm(enumerate(self._case_loader)):
            self._label_creator.create_label(case)

    def process_parallel(self, num_threads=4):
        with Pool(processes=num_threads) as pool, tqdm(total=len(self._case_loader)) as pbar:
            for _ in pool.imap(self._process_one_item_parallel, self._case_loader):
                pbar.update()
                pbar.refresh()
