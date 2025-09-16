import json
import os

import nibabel as nib
import numpy as np
import trimesh
from networkx.readwrite import json_graph
from scipy.spatial import cKDTree
from skimage import measure

from sparselabel.constants import Contours, Endings, ENCODING
from sparselabel.data_handlers.cross_section import CrossSection
from sparselabel.data_handlers.mask_image import homogenous, de_homgenize

from scipy.ndimage import binary_dilation


class CrossSectionReader:
    def __init__(self, raw_cross_section):
        self._raw_cross_section = raw_cross_section

    @property
    def lumen_contour_points(self):
        return self._read_points(Contours.INNER)

    @property
    def wall_contour_points(self):
        return self._read_points(Contours.OUTER)

    def _read_points(self, key):
        if self._raw_cross_section.get(key) is None or len(self._raw_cross_section.get(key)) == 0:
            return None
        points = np.array(self._raw_cross_section[key])
        if points.shape[0] < 3:
            raise ValueError("A contour with less than three points has no surface and cannot be processed")
        if points.shape[1] != 3:
            raise ValueError("The contour points need to be in 3D world coordinates")
        return points

    @property
    def ending_normal(self):
        if self._raw_cross_section.get(Contours.ENDING_NORMAL) is None:
            return None
        return np.array(self._raw_cross_section[Contours.ENDING_NORMAL]).reshape(-1, 1)


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
    def channel_image(self):
        return self._case.channel_image

    def load(self):
        self._case.load()
        self._load_prediction()

    def _load_prediction(self):
        file_name = self.case_id + Endings.NIFTI
        prediction_path = os.path.join(self.dataset_config.prediction_path, file_name)
        self.prediction = nib.load(prediction_path)

    @property
    def lumen_mesh_tree(self):
        if self.__lumen_mesh_tree is None:
            np.random.seed(1337)
            self.__lumen_mesh_tree = cKDTree(self.lumen_mesh.sample(10000000))
        return self.__lumen_mesh_tree

    @property
    def outer_mesh_tree(self):
        if self.__outer_mesh_tree is None:
            np.random.seed(1337)
            self.__outer_mesh_tree = cKDTree(self.outer_mesh.sample(10000000))
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
            if cross_section.lumen_points is not None:
                lumen_points.append(cross_section.lumen_points)
        if lumen_points:
            return np.vstack(lumen_points)
        return None

    def _filter_for_points_in_image(self, points):
        points_h = homogenous(points)
        points_v = de_homgenize(points_h @ np.linalg.inv(self._case.affine).T)

        points_bigger_zero = np.min(points_v, axis=1) > 0
        points_smaller_shape = np.min(self._case.image_shape[:3] - points_v - 1, axis=1) > 0

        valid_points = np.nonzero(np.logical_and(points_bigger_zero, points_smaller_shape))

        return points[valid_points]

    def _cross_section_completely_inside_volume(self, cross_section):
        contour_points_h = homogenous(cross_section.all_contour_points)
        contour_points_v = de_homgenize(contour_points_h @ np.linalg.inv(self._case.affine).T)

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
        points_h = homogenous(self.all_centerline_points())
        points_v = np.round(de_homgenize(points_h @ np.linalg.inv(self.prediction.affine).T)).astype(np.int16)
        inside_lumen = np.sum(self.prediction_volume[points_v[:, 0], points_v[:, 1], points_v[:, 2]] == self.dataset_config.lumen_value)
        return inside_lumen / points_v.shape[0]

    def _get_mesh(self, label_values):
        binary_prediction = np.isin(self.prediction_volume, label_values)
        verts, faces, normals, _ = measure.marching_cubes(binary_prediction, level=0.5)

        verts_h = homogenous(verts)
        verts_w = de_homgenize(verts_h @ self.prediction.affine.T)

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

        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = 1
        structure[0, 1, 1] = 1
        structure[1, 0, 1] = 1
        structure[1, 1, 0] = 1
        structure[2, 1, 1] = 1
        structure[1, 2, 1] = 1
        structure[1, 1, 2] = 1

        dilated_lumen = binary_dilation(lumen, structure=structure)

        touching_background = dilated_lumen & background
        touching = dilated_lumen & non_lumen

        self._lumen_background_percentage = np.sum(touching_background) / np.sum(touching)


class Case:
    def __init__(self, case_id, dataset_config):
        self.case_id = case_id
        self.dataset_config = dataset_config
        self.image = None
        self.channel_image = None
        self.cross_sections = []
        self.centerline = None

    def load(self):
        self._load_image()
        self._load_channel_image()
        self._load_cross_sections()
        self._load_centerline()

    def _load_cross_sections(self):
        raw_cross_sections = self._load_raw_cross_sections()
        self.cross_sections = []
        for identifier, raw_cross_section in raw_cross_sections.items():
            reader = CrossSectionReader(raw_cross_section)
            if reader.lumen_contour_points is None:
                continue
            self.cross_sections.append(
                CrossSection(self.dataset_config, identifier, reader.lumen_contour_points, reader.wall_contour_points, ending_normal=reader.ending_normal))

    def _load_raw_cross_sections(self):
        file_name = self.case_id + Endings.JSON
        contour_path = os.path.join(self.dataset_config.contours_path, file_name)
        with open(contour_path, "r", encoding=ENCODING) as file:
            contours = json.load(file)
        return contours

    def _load_image(self):
        file_name = self.case_id + Endings.CHANNEL_ZERO + Endings.NIFTI
        image_path = os.path.join(self.dataset_config.images_path, file_name)
        self.image = nib.load(image_path)

    def _load_channel_image(self):
        images = []

        for channel in self.dataset_config.channels.keys():
            file_name = self.case_id + "_" + f"{int(channel):04d}" + Endings.NIFTI
            image_path = os.path.join(self.dataset_config.images_path, file_name)
            images.append(nib.load(image_path).get_fdata().squeeze())

        self.channel_image = np.stack(images)

    def _load_centerline(self):
        file_name = self.case_id + Endings.JSON
        centerline_path = os.path.join(self.dataset_config.centerlines_path, file_name)
        with open(centerline_path, "r", encoding=ENCODING) as file:
            centerline_raw = json.load(file)
        self.centerline = json_graph.node_link_graph(centerline_raw, link="edges")

    @property
    def image_shape(self):
        return self.image.shape

    @property
    def affine(self):
        return self.image.affine

    @property
    def voxel_size(self):
        return self.image.header['pixdim'][1:4]

    def min_lumen_centerline_distance(self):
        centerline_points = self.all_centerline_points()
        lumen_points = self._all_lumen_points()
        if not centerline_points.size or not lumen_points.size:
            raise ValueError()
        centerline_tree = cKDTree(centerline_points)
        distances, _ = centerline_tree.query(lumen_points, k=1)
        return min(distances)

    def max_contour_centerline_distance(self):
        center_points = np.vstack([cross_section.plane_center for cross_section in self.cross_sections])
        contour_points = self._all_contour_points()

        if not center_points.size or not contour_points.size:
            raise ValueError
        centerline_tree = cKDTree(center_points)
        distances, _ = centerline_tree.query(contour_points, k=1)
        return max(distances)

    def _all_contour_points(self):
        contour_points = [np.zeros((0, 3))]
        for cross_section in self.cross_sections:
            contour_points.append(cross_section.all_contour_points)
        return np.vstack(contour_points)

    def _all_lumen_points(self):
        lumen_points = [np.zeros((0, 3))]
        for cross_section in self.cross_sections:
            lumen_points.append(cross_section.lumen_points)
        return np.vstack(lumen_points)

    def all_centerline_points(self):
        centerline_points = [np.zeros((0, 3))]
        for start, end in self.centerline.edges():
            centerline_points.append(self.centerline[start][end]['skeletons'])
        return np.vstack(centerline_points)
