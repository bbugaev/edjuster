#! /usr/bin/env python2

import os.path as path
import sys

import click
import numpy as np
from scipy.ndimage import imread

from geometry import load_mesh
from gui import run_gui


def load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.pass_context
def edjust(ctx, input_folder):
    """Adjusts pose of 3D object"""

    mesh = load_file(input_folder, 'mesh.obj', '3D object', load_mesh)

    image = load_file(input_folder, 'image.bmp', '3D object image', imread)

    model = load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    view = load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    proj = load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)

    ctx.exit(run_gui(sys.argv[:1], image))


if __name__ == '__main__':
    edjust()  # pylint: disable=no-value-for-parameter
