import itertools as itt

import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import RectBivariateSpline


def _select_points(lines, point_count):
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    total_length = lengths.sum()
    step = total_length / (point_count + 1)

    points = []
    line_indices = []
    current_length = step
    prefix_length = 0

    for i, length, line in itt.izip(itt.count(), lengths, lines):
        while current_length <= prefix_length + length:
            point_coef = (current_length - prefix_length) / length
            points.append(line[0] + (line[1] - line[0]) * point_coef)
            line_indices.append(i)
            current_length += step
        prefix_length += length

    return np.array(line_indices[:point_count]), np.array(points[:point_count])


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

    def __getitem__(self, coordinates):
        if coordinates.ndim > 1:
            coordinates = np.hsplit(coordinates[:, ::-1], 2)
        else:
            coordinates = coordinates[::-1]
        return np.hstack((
            self._x_deriv(*coordinates, grid=False),
            self._y_deriv(*coordinates, grid=False)
        ))


def _calc_normals(lines):
    normals = lines[:, 1] - lines[:, 0]
    norms = np.repeat(np.linalg.norm(normals, axis=1), 2).reshape(-1, 2)
    normals /= norms
    normals = np.dot(normals, np.array([[0, -1], [1, 0]]))
    return normals


def approximate_edge_integral(image, mesh_edges, point_count):
    edges = np.vstack((mesh_edges.borders, mesh_edges.sharp_edges))
    lines = mesh_edges.projected_vertices[edges]

    line_indices, selected_points = _select_points(lines, point_count)

    mask = (selected_points >= (0, 0)) & (selected_points < image.shape[::-1])
    mask = mask[:, 0] & mask[:, 1]
    line_indices = line_indices[mask]
    selected_points = selected_points[mask]
    lines = lines[line_indices]

    normals = _calc_normals(lines)
    gradients = Gradient(image)[selected_points]
    integral = ((normals * gradients).sum(axis=1)**2).sum()

    return integral, selected_points
