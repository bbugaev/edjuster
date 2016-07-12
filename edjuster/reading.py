from functools import partial
import os.path as path

import click
import numpy as np
from scipy.ndimage import imread

from geometry import load_mesh, Scene, Position


def _load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


def read_input(input_folder):
    rgb = _load_file(input_folder, 'image.png', '3D object image',
                     partial(imread, mode='RGB'))
    gray = _load_file(input_folder, 'image.png', '3D object image',
                      partial(imread, mode='F'))
    mesh = _load_file(input_folder, 'mesh.obj', '3D object', load_mesh)
    proj = _load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)

    model = _load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    model = Position(model.flatten())

    view = _load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    view = Position(view.flatten())

    return rgb, gray / 255.0, Scene(mesh, model, view, proj)
