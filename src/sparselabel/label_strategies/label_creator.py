import os

import numpy as np
import nibabel as nib

from sparselabel.constants import Endings
from sparselabel.data_handlers.mask_image import SparseMaskImage


class LabelCreator:
    def __init__(self, strategies, dataset_config):
        self._strategies = strategies
        self._dataset_config = dataset_config

    def apply(self, case):
        mask = SparseMaskImage(case.image_shape, case.affine, self._dataset_config)
        for strategy in self._strategies:
            strategy.apply(mask, case)
        self._save_label(mask.mask, case)

    @staticmethod
    def _save_label(mask, case):
        file_name = case.case_id + Endings.NIFTI
        labels_path = case.dataset_config.labels_path
        out_image = nib.Nifti1Image(np.astype(mask, np.int16), case.affine)
        os.makedirs(labels_path, exist_ok=True)
        nib.save(out_image, os.path.join(labels_path, file_name))
