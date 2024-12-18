import os
from glob import glob

from sparselabel.constants import Endings


class CaseLoader:
    def __init__(self, dataset_config, case_type, **kwargs):
        self.case_ids = []
        self._dataset_config = dataset_config
        self._search_cases()
        self.index = 0
        self._case_type = case_type
        self.kwargs = kwargs

    def _search_cases(self):
        self.case_ids = []
        file_name_search = "*" + Endings.CHANNEL_ZERO + Endings.NIFTI
        for path in glob(os.path.join(self._dataset_config.images_path, file_name_search)):
            case_id = os.path.basename(path)[:-len(Endings.CHANNEL_ZERO) - len(Endings.NIFTI)]
            self.case_ids.append(case_id)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.case_ids):
            case_id = self.case_ids[self.index]
            self.index += 1
            case = self._case_type(case_id, self._dataset_config, **self.kwargs)
            case.load()
            return case
        raise StopIteration

    def __len__(self):
        return len(self.case_ids)
