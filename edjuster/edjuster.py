#! /usr/bin/env python2

import os.path as path
import sys

import click
import numpy as np
from scipy.ndimage import imread

from geometry import detect_mesh_edges, load_mesh, Scene
from gui import run_gui


def load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


def load_input(input_folder):
    image = load_file(input_folder, 'image.png', '3D object image', imread)
    mesh = load_file(input_folder, 'mesh.obj', '3D object', load_mesh)
    model = load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    view = load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    proj = load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)
    return image, Scene(mesh, model, view, proj)


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.pass_context
def edjust(ctx, input_folder):
    """Adjust pose of 3D object"""

    image, scene = load_input(input_folder)
    borders, sharp_edges = detect_mesh_edges(scene)

    ctx.exit(run_gui(sys.argv[:1], image, scene))


if __name__ == '__main__':
    edjust()  # pylint: disable=no-value-for-parameter
