import os

import nibabel as nib
import numpy as np
import trimesh
import scipy.ndimage
from scipy.spatial import cKDTree
from skimage import measure

from sparselabel.constants import Endings, Evaluation
from sparselabel.data_handlers.case import Case
from sparselabel.utils import transform_points


class EvaluationCase:  # pylint: disable=too-many-instance-attributes
    def __init__(self, case_id, dataset_config):
        self._centerline_sensitivity = None
        self._lumen_background_percentage = None
        self.case_id = case_id
        self.dataset_config = dataset_config
        self.prediction = None
        self._case = Case(case_id, self.dataset_config)
        self.__lumen_mesh = None
        self.__outer_mesh = None
        self.__lumen_mesh_tree = None
        self.__outer_mesh_tree = None

    @property
    def prediction_volume(self):
        return self.prediction.get_fdata().squeeze()

    @property
    def image_data(self):
        return self._case.image_data

    def load(self):
        self._case.load()
        self._load_prediction()

    def _load_prediction(self):
        file_name = self.case_id + Endings.NIFTI
        prediction_path = os.path.join(self.dataset_config.prediction_path, file_name)
        self.prediction = nib.load(prediction_path)

    @property
    def inner_mesh_tree(self):
        # sample surface with high density for fast contour surface distance computation
        if self.__lumen_mesh_tree is None:
            np.random.seed(Evaluation.SEED)
            self.__lumen_mesh_tree = cKDTree(self.lumen_mesh.sample(Evaluation.SURFACE_SAMPLING))
        return self.__lumen_mesh_tree

    @property
    def outer_mesh_tree(self):
        # sample surface with high density for fast contour surface distance computation
        if self.__outer_mesh_tree is None:
            np.random.seed(1337)
            self.__outer_mesh_tree = cKDTree(self.outer_mesh.sample(Evaluation.SURFACE_SAMPLING))
        return self.__outer_mesh_tree

    @property
    def lumen_mesh(self):
        if self.__lumen_mesh is None:
            self.__lumen_mesh = self._get_mesh([self.dataset_config.lumen_value])
        return self.__lumen_mesh

    @property
    def outer_mesh(self):
        if self.__outer_mesh is None:
            self.__outer_mesh = self._get_mesh([self.dataset_config.lumen_value, self.dataset_config.wall_value])
        return self.__outer_mesh

    @property
    def cross_sections(self):
        return [cross_section for cross_section in self._case.cross_sections if self._cross_section_completely_inside_volume(cross_section)]

    def true_outer_wall_points(self):
        outer_wall_points = []
        for cross_section in self.cross_sections:
            if cross_section.outer_wall_points is not None:
                outer_wall_points.append(cross_section.outer_wall_points)
        if outer_wall_points:
            return np.vstack(outer_wall_points)
        return None

    def true_lumen_points(self):
        lumen_points = []
        for cross_section in self.cross_sections:
            if cross_section.inner_contour_points is not None:
                lumen_points.append(cross_section.inner_contour_points)
        if lumen_points:
            return np.vstack(lumen_points)
        return None

    def _filter_for_points_in_image(self, points):
        points_v = transform_points(points, np.linalg.inv(self._case.affine))

        points_bigger_zero = np.min(points_v, axis=1) > 0
        points_smaller_shape = np.min(self._case.image_shape[:3] - points_v - 1, axis=1) > 0

        valid_points = np.nonzero(np.logical_and(points_bigger_zero, points_smaller_shape))

        return points[valid_points]

    def _cross_section_completely_inside_volume(self, cross_section):
        contour_points_v = transform_points(cross_section.all_contour_points, np.linalg.inv(self._case.affine))

        minimum_voxel_position = np.min(contour_points_v, axis=0)
        maximum_voxel_position = np.max(contour_points_v, axis=0)

        if np.min(minimum_voxel_position) < 0:
            return False

        if np.min(self._case.image_shape[:3] - maximum_voxel_position - 1) < 0:
            return False

        return True

    @property
    def centerline(self):
        return self._case.centerline

    def all_centerline_points(self):
        return self._filter_for_points_in_image(self._case.all_centerline_points())

    @property
    def centerline_sensitivity(self):
        if self._centerline_sensitivity is None:
            self._centerline_sensitivity = self._calculate_centerline_sensitivity()
        return self._centerline_sensitivity

    def _calculate_centerline_sensitivity(self):
        points_v = transform_points(self.all_centerline_points(), np.linalg.inv(self._case.affine))
        points_v = np.round(points_v).astype(np.int16)
        inside_lumen = np.sum(self.prediction_volume[points_v[:, 0], points_v[:, 1], points_v[:, 2]] == self.dataset_config.lumen_value)
        return inside_lumen / points_v.shape[0]

    def _get_mesh(self, label_values):
        binary_prediction = np.isin(self.prediction_volume, label_values)
        verts, faces, normals, _ = measure.marching_cubes(binary_prediction, level=0.5)

        verts_w = transform_points(verts, self.prediction.affine)

        return trimesh.Trimesh(vertices=verts_w, faces=faces, vertex_normals=normals)

    @property
    def lumen_background_percentage(self):
        if self._lumen_background_percentage is None:
            self._compute_lumen_background_percentage()
        return self._lumen_background_percentage

    def _compute_lumen_background_percentage(self):
        lumen = self.prediction_volume == self.dataset_config.lumen_value
        non_lumen = self.prediction_volume != self.dataset_config.lumen_value
        background = self.prediction_volume == self.dataset_config.background_value

        neighborhood_6 = np.zeros((3, 3, 3), dtype=bool)
        neighborhood_6[1, 1, 1] = 1
        neighborhood_6[0, 1, 1] = 1
        neighborhood_6[1, 0, 1] = 1
        neighborhood_6[1, 1, 0] = 1
        neighborhood_6[2, 1, 1] = 1
        neighborhood_6[1, 2, 1] = 1
        neighborhood_6[1, 1, 2] = 1

        dilated_lumen = scipy.ndimage.binary_dilation(lumen, structure=neighborhood_6)

        touching_background = dilated_lumen & background
        touching = dilated_lumen & non_lumen

        self._lumen_background_percentage = np.sum(touching_background) / np.sum(touching)
