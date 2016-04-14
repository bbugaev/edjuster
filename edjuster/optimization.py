from itertools import izip

import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import RectBivariateSpline


def select_points(mesh_edges, point_count):
    edges = np.vstack((mesh_edges.borders, mesh_edges.sharp_edges))
    vertices = np.hsplit(mesh_edges.projected_vertices[edges], 2)

    lengths = np.linalg.norm(vertices[1] - vertices[0], axis=2)
    lengths.shape = lengths.shape[:1]
    total_length = lengths.sum()
    step = total_length / (point_count + 1)

    vertices = np.hstack(vertices)

    points = []
    current_length = step
    prefix_length = 0

    for length, line in izip(lengths, vertices):
        while current_length <= prefix_length + length:
            point_coef = (current_length - prefix_length) / length
            points.append(line[0] + (line[1] - line[0]) * point_coef)
            current_length += step
        prefix_length += length

    return np.array(points[:point_count])


class Gradient(object):
    SCHARR_KERNEL_X = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 32.0
    SCHARR_KERNEL_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) / 32.0

    def __init__(self, image):
        image = np.flipud(image)
        x_indices = np.arange(image.shape[1])
        y_indices = np.arange(image.shape[0])
        x_deriv = convolve(image, Gradient.SCHARR_KERNEL_X)
        y_deriv = convolve(image, Gradient.SCHARR_KERNEL_Y)
        self._x_deriv = RectBivariateSpline(y_indices, x_indices, x_deriv)
        self._y_deriv = RectBivariateSpline(y_indices, x_indices, y_deriv)

    def __call__(self, coordinates):
        if coordinates.ndim > 1:
            coordinates = np.hsplit(coordinates[:, ::-1], 2)
        else:
            coordinates = coordinates[::-1]
        return np.hstack((
            self._x_deriv(*coordinates, grid=False),
            self._y_deriv(*coordinates, grid=False)
        ))


def approximate_edge_integral(image, mesh_edges, point_count):
    pass
