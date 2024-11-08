import os

import numpy as np
import nibabel as nib

from constants import Labels, Endings, data_raw, Folders
from mask_image import SparseMaskImage


def _save_label(mask, case):
    file_name = case.case_id + Endings.NIFTI
    labels_path = os.path.join(data_raw, case.dataset, Folders.LABELS)
    out_image = nib.Nifti1Image(np.astype(mask, np.int16), case.affine)
    os.makedirs(labels_path, exist_ok=True)
    nib.save(out_image, os.path.join(labels_path, file_name))


class LabelCreator:
    def __init__(self, strategies):
        self._strategies = strategies

    def create_label(self, case):
        mask = SparseMaskImage(case.image_shape, case.affine)
        for strategy in self._strategies:
            strategy.apply(mask, case)
        _save_label(mask.mask, case)

