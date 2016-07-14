#! /usr/bin/env python2

import click
import matplotlib.pyplot as plt
import numpy as np

from geometry import Position
from reading import read_input
from optimization import IntegralCalculator


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
def plot(input_folder):
    _, gray, scene = read_input(input_folder)
    integral_calculator = IntegralCalculator(gray, scene,
                                             normalized_gradient=True)

    center_point = scene.model.vector6
    point_count = 200
    deltas = [20] * 3 + [30] * 3
    x_labels = [r'$x$', r'$y$', r'$z$', r'$\alpha$', r'$\beta$', r'$\gamma$']

    for axis in xrange(6):
        points = np.tile(center_point, (point_count, 1))
        limits = (center_point[axis] - deltas[axis],
                  center_point[axis] + deltas[axis])
        points[:, axis] = np.linspace(limits[0], limits[1], point_count)
        values = np.array([integral_calculator(Position(p)) for p in points])

        plt.subplot(231 + axis)
        plt.xlabel(x_labels[axis], fontsize='x-large')
        plt.grid(True)
        plt.xlim(*limits)
        plt.gca().xaxis.set_major_locator(plt.LinearLocator(3))
        plt.plot(points[:, axis], values)

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.show()


if __name__ == '__main__':
    plot()  # pylint: disable=no-value-for-parameter
