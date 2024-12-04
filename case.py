import json
import os
from glob import glob

import nibabel as nib
import numpy as np
import trimesh
from networkx.readwrite import json_graph
from scipy.spatial import cKDTree
from skimage import measure

from constants import Contours, Endings, data_raw, Folders, Labels, data_results
from cross_section import CrossSection
from mask_image import homogenous, de_homgenize


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


class EvaluationCase:
    def __init__(self, case_id, dataset, prediction_path=None):
        self._centerline_sensitivity = None
        self.case_id = case_id
        self.dataset = dataset
        self.prediction = None
        if prediction_path is None:
            self.prediction_path = os.path.join(data_results, self.dataset, Folders.FULLRES_TRAINER, Folders.CROSS_VALIDATION_RESULTS, Folders.POSTPROCESSED)
        else: 
            self.prediction_path = prediction_path
        self._case = Case(case_id, dataset)
        self.__lumen_mesh = None
        self.__outer_mesh = None

    def load(self):
        self._case.load()
        self._load_prediction()

    def _load_prediction(self):
        file_name = self.case_id + Endings.NIFTI
        prediction_path = os.path.join(self.prediction_path, file_name)
        self.prediction = nib.load(prediction_path)

    @property
    def lumen_mesh(self):
        if self.__lumen_mesh is None:
            self.__lumen_mesh = self._get_mesh(self.prediction, [Labels.LUMEN])
        return self.__lumen_mesh

    @property
    def outer_mesh(self):
        if self.__outer_mesh is None:
            self.__outer_mesh = self._get_mesh(self.prediction, [Labels.LUMEN, Labels.WALL])
        return self.__outer_mesh

    @property
    def cross_sections(self):
        return self._case.cross_sections

    @property
    def centerline(self):
        return self._case.centerline

    def all_centerline_points(self):
        return self._case.all_centerline_points()
    
    @property
    def centerline_sensitivity(self):
        if self._centerline_sensitivity is None:
            self._centerline_sensitivity = self._calculate_centerline_sensitivity()
        return self._centerline_sensitivity
    
    def _calculate_centerline_sensitivity(self):
        inside_mesh = self.lumen_mesh.contains(self.all_centerline_points())
        return np.sum(inside_mesh) / len(inside_mesh)

    def _get_mesh(self, prediction, label_values):
        binary_prediction = np.isin(prediction.get_fdata(), label_values)
        verts, faces, normals, _ = measure.marching_cubes(binary_prediction, level=0.5)

        verts_h = homogenous(verts)
        verts_w = de_homgenize(verts_h @ prediction.affine.T)

        return trimesh.Trimesh(vertices=verts_w, faces=faces, vertex_normals=normals)


class Case:
    def __init__(self, case_id, dataset):
        self.case_id = case_id
        self.dataset = dataset
        self.image = None
        self.cross_sections = []
        self.centerline = None

    def load(self):
        self._load_image()
        self._load_cross_sections()
        self._load_centerline()

    def _load_cross_sections(self):
        raw_cross_sections = self._load_raw_cross_sections()
        self.cross_sections = []
        for identifier, raw_cross_section in raw_cross_sections.items():
            reader = CrossSectionReader(raw_cross_section)
            if reader.lumen_contour_points is None:
                continue
            self.cross_sections.append(CrossSection(identifier, reader.lumen_contour_points, reader.wall_contour_points, ending_normal=reader.ending_normal))

    def _load_raw_cross_sections(self):
        file_name = self.case_id + Endings.JSON
        contour_path = os.path.join(data_raw, self.dataset, Folders.CONTOURS, file_name)
        with open(contour_path, "r") as file:
            contours = json.load(file)
        return contours

    def _load_image(self):
        file_name = self.case_id + Endings.CHANNEL_ZERO + Endings.NIFTI
        image_path = os.path.join(data_raw, self.dataset, Folders.IMAGES, file_name)
        self.image = nib.load(image_path)

    def _load_centerline(self):
        file_name = self.case_id + Endings.JSON
        centerline_path = os.path.join(data_raw, self.dataset, Folders.CENTERLINES, file_name)
        with open(centerline_path, "r") as file:
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


class CaseLoader:
    def __init__(self, dataset, case_type=Case):
        self.case_ids = []
        self._dataset = dataset
        self._search_cases()
        self.index = 0
        self._case_type = case_type

    def _search_cases(self):
        self.case_ids = []
        file_name_search = "*" + Endings.CHANNEL_ZERO + Endings.NIFTI
        for path in glob(os.path.join(data_raw, self._dataset, Folders.IMAGES, file_name_search)):
            case_id = os.path.basename(path)[:-len(Endings.CHANNEL_ZERO) - len(Endings.NIFTI)]
            self.case_ids.append(case_id)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.case_ids):
            case_id = self.case_ids[self.index]
            self.index += 1
            case = self._case_type(case_id, self._dataset)
            case.load()
            return case
        else:
            raise StopIteration

    def __len__(self):
        return len(self.case_ids)
