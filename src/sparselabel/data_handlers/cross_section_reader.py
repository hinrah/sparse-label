import numpy as np

from sparselabel.constants import Contours


class CrossSectionReader:
    def __init__(self, raw_cross_section):
        self._raw_cross_section = raw_cross_section

    @property
    def inner_contour_points(self):
        return self._read_contour_points(Contours.INNER)

    @property
    def outer_contour_points(self):
        return self._read_contour_points(Contours.OUTER)

    def _read_contour_points(self, key):
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
