from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from networkx.readwrite import json_graph
from scipy.spatial import cKDTree

from sparselabel.data_handlers.cross_section_reader import CrossSectionReader
from sparselabel.constants import Endings, ENCODING
from sparselabel.data_handlers.cross_section import CrossSection
from sparselabel.dataset_config import DatasetConfig


class Case:
    def __init__(self, case_id: str, dataset_config: DatasetConfig) -> None:
        self.case_id = case_id
        self.dataset_config = dataset_config
        self._image_shape = None
        self._image_affine = None
        self._voxel_size = None
        self.image_data = None
        self.cross_sections = []
        self.centerline = None

    def load(self) -> None:
        self._load_image_properties()
        self._load_image()
        self._load_cross_sections()
        self._load_centerline()

    def _load_cross_sections(self) -> None:
        raw_cross_sections = self._load_raw_cross_sections()
        self.cross_sections = []
        for identifier, raw_cross_section in raw_cross_sections.items():
            reader = CrossSectionReader(raw_cross_section)
            if reader.inner_contour_points is None:
                continue
            self.cross_sections.append(
                CrossSection(self.dataset_config, identifier, reader.inner_contour_points, reader.outer_contour_points, ending_normal=reader.ending_normal))

    def _load_raw_cross_sections(self) -> dict:
        file_name = self.case_id + Endings.JSON
        contour_path = os.path.join(self.dataset_config.contours_path, file_name)
        with open(contour_path, "r", encoding=ENCODING) as file:
            contours = json.load(file)
        return contours

    def _load_image_properties(self) -> None:
        file_name = self.case_id + Endings.CHANNEL_ZERO + Endings.NIFTI
        image_path = os.path.join(self.dataset_config.images_path, file_name)
        image = nib.load(image_path)
        self._image_shape = image.shape
        self._image_affine = image.affine
        self._voxel_size = image.header['pixdim'][1:4]

    def _load_image(self) -> None:
        images = []

        for channel in self.dataset_config.channels.keys():
            file_name = self.case_id + "_" + f"{int(channel):04d}" + Endings.NIFTI
            image_path = os.path.join(self.dataset_config.images_path, file_name)
            images.append(nib.load(image_path).get_fdata().squeeze())

        self.image_data = np.stack(images)

    def _load_centerline(self) -> None:
        file_name = self.case_id + Endings.JSON
        centerline_path = os.path.join(self.dataset_config.centerlines_path, file_name)
        with open(centerline_path, "r", encoding=ENCODING) as file:
            centerline_raw = json.load(file)
        self.centerline = json_graph.node_link_graph(centerline_raw, link="edges")

    @property
    def image_shape(self) -> tuple | None:
        return self._image_shape

    @property
    def affine(self) -> np.ndarray | None:
        return self._image_affine

    @property
    def voxel_size(self) -> np.ndarray | None:
        return self._voxel_size

    def max_contour_centerline_distance(self) -> float:
        center_points = np.vstack([cross_section.plane_center for cross_section in self.cross_sections])
        contour_points = self._all_contour_points()

        if not center_points.size or not contour_points.size:
            raise ValueError
        centerline_tree = cKDTree(center_points)
        distances, _ = centerline_tree.query(contour_points, k=1)
        return max(distances)

    def _all_contour_points(self) -> np.ndarray:
        contour_points = [np.zeros((0, 3))]
        for cross_section in self.cross_sections:
            contour_points.append(cross_section.all_contour_points)
        return np.vstack(contour_points)

    def all_inner_contour_points(self) -> np.ndarray:
        lumen_points = [np.zeros((0, 3))]
        for cross_section in self.cross_sections:
            lumen_points.append(cross_section.inner_contour_points)
        return np.vstack(lumen_points)

    def all_centerline_points(self) -> np.ndarray:
        centerline_points = [np.zeros((0, 3))]
        for start, end in self.centerline.edges():
            centerline_points.append(self.centerline[start][end]['skeletons'])
        return np.vstack(centerline_points)
