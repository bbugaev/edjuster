#! /usr/bin/env python2

import os.path as path

import click
import numpy as np
from scipy.ndimage import imread

from geometry import load_mesh


def load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
def edjust(input_folder):
    """Adjusts pose of 3D object"""

    mesh = load_file(input_folder, 'mesh.obj', '3D object', load_mesh)

    image = load_file(input_folder, 'image.bmp', '3D object image', imread)

    model = load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    view = load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    proj = load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)


if __name__ == '__main__':
    edjust()  # pylint: disable=no-value-for-parameter
