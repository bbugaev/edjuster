import itertools as itt

import numpy as np
from scipy.ndimage import convolve, sobel
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import basinhopping

from geometry import MeshEdgeDetector, Position


class Gradient(object):
    SCHARR = 'Scharr'
    SOBEL = 'Sobel'

    SCHARR_KERNEL_X = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 32.0
    SCHARR_KERNEL_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) / 32.0

    def __init__(self, image, method=SOBEL):
        if method not in (Gradient.SCHARR, Gradient.SOBEL):
            raise ValueError(
                'Invalid method (use Gradient.SCHARR or Gradient.SOBEL)'
            )
        image = np.flipud(image)
        x_indices = np.arange(image.shape[1])
        y_indices = np.arange(image.shape[0])
        if method == Gradient.SCHARR:
            x_deriv = convolve(image, Gradient.SCHARR_KERNEL_X)
            y_deriv = convolve(image, Gradient.SCHARR_KERNEL_Y)
        else:
            x_deriv = sobel(image, axis=1) / 8.0
            y_deriv = sobel(image, axis=0) / 8.0
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


class IntegralCalculator(object):
    DEFAULT_POINT_COUNT = 1000

    def __init__(self, image, scene, point_count=DEFAULT_POINT_COUNT):
        self._image_size = image.shape
        self._scene = scene
        self._point_count = point_count

        self._gradient = Gradient(image)
        self._mesh_edge_detector = MeshEdgeDetector(scene.mesh)

    def __call__(self, position):
        _, _, gradients, normals = self.calc_gradients_and_normals(position)

        integral = ((normals * gradients).sum(axis=1)**2).sum()
        integral /= normals.shape[0]

        return integral

    def calc_gradients_and_normals(self, position):
        scene = self._scene._replace(model=position)
        mesh_edges = self._mesh_edge_detector(
            scene.proj.dot(scene.view.matrix.dot(scene.model.matrix)),
            self._image_size
        )

        edges = np.vstack((mesh_edges.borders, mesh_edges.sharp_edges))
        lines = mesh_edges.projected_vertices[edges]

        points, line_indices = self._select_points(lines)
        lines = lines[line_indices]

        normals = IntegralCalculator._calc_normals(lines)
        gradients = self._gradient[points]

        return mesh_edges, points, gradients, normals

    def _select_points(self, lines):
        lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
        total_length = lengths.sum()
        step = total_length / (self._point_count + 1)

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

        return self._clip(
            np.array(points[:self._point_count]),
            np.array(line_indices[:self._point_count])
        )

    def _clip(self, points, line_indices):
        mask = (points >= (0, 0)) & (points < self._image_size[::-1])
        mask = mask[:, 0] & mask[:, 1]
        return points[mask], line_indices[mask]

    @staticmethod
    def _calc_normals(lines):
        normals = lines[:, 1] - lines[:, 0]
        norms = np.repeat(np.linalg.norm(normals, axis=1), 2).reshape(-1, 2)
        normals /= norms
        normals = np.dot(normals, np.array([[0, -1], [1, 0]]))
        return normals


class Walker(object):

    def __init__(self, step_vector, step_size):
        self.step_vector = step_vector
        self.stepsize = step_size

    def __call__(self, vector):
        random_vector = np.random.uniform(-1, 1, vector.shape)
        return vector + self.step_vector * random_vector * self.stepsize


def optimize_model(model, integral_calculator, step_callback, echo):
    walker = Walker(
        np.array([5, 5, 5, 1, 1, 1]),
        0.5
    )
    minimizer_kwargs = {
        'method': 'SLSQP',
        'tol': 0.00001,
        'options': {
            'maxiter': 50,
            'disp': echo
        }
    }
    basinhopping_result = basinhopping(
        lambda x: 10 - 1000 * integral_calculator(Position(x)),
        model.vector6,
        niter=100,
        T=0.5,
        minimizer_kwargs=minimizer_kwargs,
        take_step=walker,
        callback=step_callback,
        disp=echo
    )

    if echo:
        print
        print basinhopping_result

    return Position(basinhopping_result.x)
