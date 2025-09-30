import numpy as np
from scipy.spatial import cKDTree


def get_max_voxel_size(cases):
    if not cases:
        raise ValueError("There are no cases to extract a voxel size from")

    max_voxel_size = 0
    for case in cases:
        max_voxel_size = max(max_voxel_size, max(case.voxel_size))  # pylint: disable=nested-min-max
    return max_voxel_size


def get_median_voxel_size(cases):
    if not cases:
        raise ValueError("There are no cases to extract a voxel size from")
    voxel_sizes = []
    for case in cases:
        voxel_sizes.append(case.voxel_size)
    return np.median(np.array(voxel_sizes).flatten())


def get_min_lumen_centerline_distance(cases):
    min_lumen_centerline_distances = []
    for case in cases:
        try:
            min_lumen_centerline_distances.append(min_lumen_centerline_distance_one_case(case))
        except ValueError:
            continue
    return np.percentile(min_lumen_centerline_distances, 5)


def min_lumen_centerline_distance_one_case(case):
    centerline_points = case.all_centerline_points()
    lumen_points = case.all_inner_contour_points()
    if not centerline_points.size or not lumen_points.size:
        raise ValueError()
    centerline_tree = cKDTree(centerline_points)
    distances, _ = centerline_tree.query(lumen_points, k=1)
    return min(distances)


def get_max_contour_centerline_distance(cases):
    max_contour_centerline_distances = []
    for case in cases:
        try:
            max_contour_centerline_distances.append(case.max_contour_centerline_distance())
        except ValueError:
            continue
    return max(max_contour_centerline_distances)
