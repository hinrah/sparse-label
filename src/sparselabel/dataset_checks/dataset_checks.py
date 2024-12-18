import networkx as nx
from networkx import DiGraph

from sparselabel.data_handlers.cross_section import ContourDoesNotExistError
from sparselabel.data_handlers.cross_section_centerline import CrossSectionCenterline


class DataErrorNotAllDataUsable(Exception):
    pass


class DataErrorThatLeadsToWrongLabels(Exception):
    pass


class CenterlineInsideLumen:
    def __init__(self):
        pass

    def check(self, case):
        for cross_section in case.cross_sections:
            centerline = CrossSectionCenterline(case.centerline, cross_section)
            if not centerline.has_valid_centerline():
                raise DataErrorThatLeadsToWrongLabels(
                    "case {} has no centerline that is inside the lumen of cross_section {}.".format(case.case_id, cross_section.identifier))


class LumenInsideWall:
    def __init__(self):
        pass

    def check(self, case):
        for cross_section in case.cross_sections:
            try:
                if not cross_section.lumen_is_inside_wall():
                    raise DataErrorNotAllDataUsable(
                        "case {} has a lumen that is not inside the wall for cross_section {}.".format(case.case_id, cross_section.identifier))
            except ContourDoesNotExistError as exc:
                raise DataErrorNotAllDataUsable("case {} cross_section {} has no wall contour.".format(case.case_id, cross_section.identifier)) from exc


class CrossSectionWithoutLumenContour:
    def __init__(self):
        pass

    def check(self, case):
        for cross_section in case.cross_sections:
            if cross_section.lumen_points is None:
                raise DataErrorNotAllDataUsable("case {} has a cross_section without a lumen contour.".format(case.case_id))


class CrossSectionWithoutWallContour:
    def __init__(self):
        pass

    def check(self, case):
        for cross_section in case.cross_sections:
            if cross_section.outer_wall_contour is None:
                raise DataErrorNotAllDataUsable("case {} has a cross_section without a wall contour.".format(case.case_id))


class NumArteries:
    def __init__(self, expected_number):
        self._expected_number = expected_number

    def check(self, case):
        if isinstance(case.centerline, DiGraph):
            centerline = case.centerline.to_undirected()
        else:
            centerline = case.centerline
        num_components = len(list(nx.connected_components(centerline)))
        if num_components != self._expected_number:
            raise DataErrorThatLeadsToWrongLabels("case {} has {} arteries. Expected number is {}.".format(case.case_id, num_components, self._expected_number))
