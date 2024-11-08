from constants import Labels
from labeling_strategies import LabelCrossSection, LabelCenterline
from mask_image import SparseMaskImage

class DefaultLabelCreator:
    def __init__(self, cross_section_thickness, min_lumen_radius, max_wall_radius):
        self.mask = None
        self._cross_section_thickness = cross_section_thickness
        self._min_lumen_radius = min_lumen_radius
        self._max_wall_radius = max_wall_radius

    def create_label(self, case):
        self.mask = SparseMaskImage(case.image_shape, case.affine)
        self._apply_cross_section_based_strategies(case)
        self._apply_centerline_based_strategies(case)
        case.save_label(self.mask.mask)

    def _apply_cross_section_based_strategies(self, case):
        for cross_section in case.cross_sections:
            label_non_plane_voxels = LabelCrossSection(cross_section, self._cross_section_thickness / 2, case.centerline)
            label_non_plane_voxels.apply(self.mask)

    def _apply_centerline_based_strategies(self, case):
        label_background_based_on_centerline = LabelCenterline(case.centerline, self._max_wall_radius, Labels.BACKGROUND)
        label_lumen_based_on_centerline = LabelCenterline(case.centerline, self._min_lumen_radius, Labels.LUMEN)
        label_lumen_based_on_centerline.apply(self.mask)
        label_background_based_on_centerline.apply(self.mask)
