import networkx as nx
from networkx import DiGraph

from sparselabel.data_handlers.case import Case
from sparselabel.data_handlers.cross_section import ContourDoesNotExistError
from sparselabel.data_handlers.cross_section_scope_classifier import CrossSectionScopeClassifier


class DataErrorNotAllDataUsable(Exception):
    pass


class DataErrorThatLeadsToWrongLabels(Exception):
    pass


class CenterlineInsideInnerContour:
    def __init__(self):
        pass

    def check(self, case):
        for cross_section in case.cross_sections:
            centerline = CrossSectionScopeClassifier(case.centerline, cross_section)
            if not centerline.has_valid_centerline():
                raise DataErrorThatLeadsToWrongLabels(
                    "case {} has no centerline that is inside the inner_contour of cross_section {}.".format(case.case_id, cross_section.identifier))


class InnerContourWithinOuter:
    def __init__(self):
        pass

    def check(self, case: Case):
        for cross_section in case.cross_sections:
            try:
                if not cross_section.inner_contour_inside_outer_contour():
                    raise DataErrorNotAllDataUsable(
                        "case {} has a inner_contour that is not inside the outer_contour for cross_section {}.".format(case.case_id, cross_section.identifier))
            except ContourDoesNotExistError as exc:
                raise DataErrorNotAllDataUsable("case {} cross_section {} has no outer contour.".format(case.case_id, cross_section.identifier)) from exc


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
