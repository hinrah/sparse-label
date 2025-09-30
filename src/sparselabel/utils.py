import numpy as np


def homogenous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def de_homogenize(points):
    return points[:, :3]


def transform_points(points, matrix):
    points_h = homogenous(points)
    return de_homogenize(points_h @ matrix.T)
