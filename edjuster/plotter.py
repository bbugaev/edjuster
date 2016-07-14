#! /usr/bin/env python2

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import pyramid_gaussian

from geometry import Position
from reading import read_input
from optimization import IntegralCalculator


def calc_pyramid(image, max_lvl, downscale=2):
    return tuple(pyramid_gaussian(image, downscale=downscale))[:max_lvl]


def blur(image, max_lvl, sigma=3):
    blurred = []
    for _ in xrange(max_lvl):
        blurred.append(image)
        image = gaussian_filter(image, sigma)
    return tuple(blurred)


def calc_translation_delta(vertices):
    delta = 0
    for axis in xrange(vertices.shape[1]):
        coords = vertices[:, axis]
        delta = max(delta, coords.max() - coords.min())
    return delta / 10


def build_plot_data(integral_calculator, center_point, axis, limits,
                    point_count):
    points = np.tile(center_point, (point_count, 1))
    varying_coords = np.linspace(limits[0], limits[1], point_count)
    points[:, axis] = varying_coords
    values = np.array([integral_calculator(Position(p)) for p in points])
    return varying_coords, values


ANGLE_DELTA_IN_DEGREES = 30
PYRAMID_LVL_COUNT = 3
X_LABELS = [r'$x$', r'$y$', r'$z$', r'$\alpha$', r'$\beta$', r'$\gamma$']


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.option('-c', '--plot-point-count', default=200)
@click.option('-d', '--points-per-pixel', default=0.3)
@click.option('-n', '--not-normalized-gradient', is_flag=True)
def plot(input_folder, plot_point_count, points_per_pixel,
         not_normalized_gradient):
    _, gray, scene = read_input(input_folder)

    pyramid = calc_pyramid(gray, PYRAMID_LVL_COUNT, 2)
    center_point = scene.model.vector6
    deltas = [calc_translation_delta(scene.mesh.vertices)] * 3 + \
             [ANGLE_DELTA_IN_DEGREES] * 3

    for image in pyramid:
        integral_calculator = IntegralCalculator(image, scene,
                                                 points_per_pixel,
                                                 not not_normalized_gradient)
        for axis in xrange(6):
            limits = (center_point[axis] - deltas[axis],
                      center_point[axis] + deltas[axis])
            plot_data = build_plot_data(integral_calculator, center_point,
                                        axis, limits, plot_point_count)
            plt.subplot(231 + axis)
            plt.grid(True)
            plt.xlabel(X_LABELS[axis], fontsize='x-large')
            plt.xlim(*limits)
            plt.gca().xaxis.set_major_locator(plt.LinearLocator(3))
            plt.plot(*plot_data)

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.show()


if __name__ == '__main__':
    plot()  # pylint: disable=no-value-for-parameter
