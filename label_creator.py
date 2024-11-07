import numpy as np

from constants import Labels
from labeling_strategies import LabelDistantVoxelsAsUnknown, LabelVoxelsBasedOnProjectionToCrossSection, LabelUnknownCrosssSectionBackground, LabelCenterline
from mask_image import SparseMaskImage


class DefaultLabelCreator:
    def __init__(self, cross_section_thickness, min_lumen_radius, max_wall_radius):
        self._cross_section_masks = []
        self._joined_mask = None
        self._cross_section_thickness = cross_section_thickness
        self._min_lumen_radius = min_lumen_radius
        self._max_wall_radius = max_wall_radius

    def _reset(self):
        self._cross_section_masks = []
        self._joined_mask = None

    def create_label(self, case):
        self._reset()
        self._apply_cross_section_based_strategies(case)
        self._join_cross_section_masks(case)
        self._apply_centerline_based_strategies(case)
        case.save_label(self._joined_mask.mask)

    def _apply_cross_section_based_strategies(self, case):
        for cross_section in case.cross_sections:
            mask = SparseMaskImage(case.image_shape, case.affine)

            label_non_plane_voxels = LabelDistantVoxelsAsUnknown(cross_section, self._cross_section_thickness/2)
            label_in_plane_voxels = LabelVoxelsBasedOnProjectionToCrossSection(cross_section)
            label_background_as_unknown_based_on_centerline = LabelUnknownCrosssSectionBackground(cross_section, case.centerline)
            mask.label_points_with_label(label_non_plane_voxels, Labels.UNPROCESSED)
            mask.label_points_with_label(label_in_plane_voxels, Labels.UNPROCESSED)
            mask.label_points_with_label(label_background_as_unknown_based_on_centerline, Labels.BACKGROUND)

            self._cross_section_masks.append(mask)

    def _join_cross_section_masks(self, case):
        if len(self._cross_section_masks) == 0:
            self._joined_mask = SparseMaskImage(case.image_shape, case.affine)
            self._joined_mask.set_mask(np.ones(case.image_shape)*Labels.UNKNOWN)
            print("Case without a cross-section. This will give limited additional info as only centerline based labels are created. Consider removing this case.")
            return

        all_masks = np.stack([mask.mask for mask in self._cross_section_masks])

        min_label = np.min(all_masks, axis=0)
        is_valid = np.all(np.logical_or(all_masks == min_label, all_masks == Labels.UNKNOWN), axis=0)

        joined_image = np.where(is_valid, min_label, Labels.UNKNOWN)

        self._joined_mask = SparseMaskImage(case.image_shape, case.affine)
        self._joined_mask.set_mask(joined_image)

    def _apply_centerline_based_strategies(self, case):
        label_lumen_based_on_centerline = LabelCenterline(case.centerline, self._min_lumen_radius, Labels.LUMEN)
        label_background_based_on_centerline = LabelCenterline(case.centerline, self._max_wall_radius, Labels.BACKGROUND)
        self._joined_mask.label_points_with_label(label_lumen_based_on_centerline, Labels.UNKNOWN)
        self._joined_mask.label_points_with_label(label_background_based_on_centerline, Labels.UNKNOWN)
