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
    integral_calculator = IntegralCalculator(gray, scene)

    center = scene.model.vector6
    point_count = 300
    axis = 4
    delta = 10

    points = np.tile(center, (point_count, 1))
    points[:, axis] = np.linspace(center[axis] - delta, center[axis] + delta,
                                  point_count)
    values = np.array([integral_calculator(Position(p)) for p in points])

    plt.plot(points[:, axis], values)
    plt.show()


if __name__ == '__main__':
    plot()  # pylint: disable=no-value-for-parameter
