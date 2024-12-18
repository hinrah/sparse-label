import json
import os
from glob import glob
from pathlib import Path

from sparselabel.constants import EnvironmentVars, Folders, DatasetInfo, ENCODING


class DatasetConfig:
    def __init__(self, dataset_name, folder_postfix="Tr",
                 prediction_sub_path=os.path.join("nnUNetTrainer__nnUNetPlans__3d_fullres", "crossval_results_folds_0_1_2_3_4")):
        self.data_raw = self._get_from_environment_variable(EnvironmentVars.sparse_vessel_masks_raw, EnvironmentVars.nnunet_raw)
        self.data_results = self._get_from_environment_variable(EnvironmentVars.sparse_vessel_masks_results, EnvironmentVars.nnunet_results)
        self._folder_postfix = folder_postfix
        self._prediction_sub_path = prediction_sub_path
        try:
            dataset_id = int(dataset_name)
            self.dataset_name = self._get_name_from_id(dataset_id)
        except (TypeError, ValueError):
            self.dataset_name = dataset_name

        self._dataset_info = None

    @property
    def dataset_info(self):
        if self._dataset_info is None:
            with open(self.dataset_info_path, encoding=ENCODING) as file:
                self._dataset_info = json.load(file)
        return self._dataset_info

    def _get_name_from_id(self, dataset_id):
        datasets = list(glob(os.path.join(self.data_raw, "Dataset{:03}_*/".format(dataset_id))))
        if len(datasets) > 1:
            raise RuntimeError(f"There are more than one dataset with id {dataset_id}")

        if len(datasets) == 0:
            raise RuntimeError(f"There are no datasets with id {dataset_id}")

        return Path(datasets[0]).name

    def _get_from_environment_variable(self, first_choice, second_choice):
        result = os.environ.get(first_choice)
        if result is None:
            print(f"{first_choice} is not defined {second_choice} is now used.")
            result = os.environ.get(second_choice)
            if result is None:
                print(f"{second_choice} is not defined as well. Sparse vessel masks can not be created.")

        return result

    @property
    def contours_path(self):
        return os.path.join(self.data_raw, self.dataset_name, Folders.CONTOURS + self._folder_postfix)

    @property
    def images_path(self):
        return os.path.join(self.data_raw, self.dataset_name, Folders.IMAGES + self._folder_postfix)

    @property
    def prediction_path(self):
        return os.path.join(self.data_results, self.dataset_name, self._prediction_sub_path)

    @property
    def centerlines_path(self):
        return os.path.join(self.data_raw, self.dataset_name, Folders.CENTERLINES + self._folder_postfix)

    @property
    def labels_path(self):
        return os.path.join(self.data_raw, self.dataset_name, Folders.LABELS + self._folder_postfix)

    @property
    def results_path(self):
        return os.path.join(self.data_results, self.dataset_name)

    @property
    def raw_path(self):
        return os.path.join(self.data_raw, self.dataset_name)

    @property
    def dataset_info_path(self):
        return os.path.join(self.data_raw, self.dataset_name, DatasetInfo.FILE_NAME)

    def class_value_by_label(self, label):
        return self.dataset_info[DatasetInfo.LABELS][label]

    @property
    def lumen_value(self):
        return self.dataset_info[DatasetInfo.LABELS][DatasetInfo.LUMEN]

    @property
    def background_value(self):
        return self.dataset_info[DatasetInfo.LABELS][DatasetInfo.BACKGROUND]

    @property
    def wall_value(self):
        return self.dataset_info[DatasetInfo.LABELS][DatasetInfo.WALL]

    @property
    def ignore_value(self):
        return self.dataset_info[DatasetInfo.LABELS][DatasetInfo.IGNORE]

    @property
    def has_wall(self):
        return DatasetInfo.WALL in self.dataset_info[DatasetInfo.LABELS]

    @property
    def classes(self):
        if self.has_wall:
            return [DatasetInfo.BACKGROUND, DatasetInfo.WALL, DatasetInfo.LUMEN]
        return [DatasetInfo.BACKGROUND, DatasetInfo.LUMEN]
