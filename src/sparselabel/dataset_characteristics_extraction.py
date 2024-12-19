import numpy as np


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
            min_lumen_centerline_distances.append(case.min_lumen_centerline_distance())
        except ValueError:
            continue
    return np.percentile(min_lumen_centerline_distances, 5)


def get_max_contour_centerline_distance(cases):
    max_contour_centerline_distances = []
    for case in cases:
        try:
            max_contour_centerline_distances.append(case.max_contour_centerline_distance())
        except ValueError:
            continue
    return max(max_contour_centerline_distances)
