#! /usr/bin/env python2

from functools import partial
from multiprocessing import Process
from multiprocessing.queues import SimpleQueue
import os.path as path
import sys

import click
import numpy as np
from scipy.ndimage import imread

from geometry import load_mesh, Scene, Position
from gui import run_gui
from optimization import optimize_model


def load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


def load_input(input_folder):
    image = load_file(input_folder, 'image.png', '3D object image',
                      partial(imread, mode='L'))
    mesh = load_file(input_folder, 'mesh.obj', '3D object', load_mesh)
    proj = load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)

    model = load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    model = Position(model[0], model[1])

    view = load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    view = Position(view[0], view[1])

    return image, Scene(mesh, model, view, proj)


def run_optimization(image, scene, model_queue):

    def process_step(vector6, _, accepted):
        if accepted:
            model_queue.put(Position(vector6[:3], vector6[3:]))

    optimized_model = optimize_model(image / 255.0, scene, process_step, True)
    model_queue.put(optimized_model)

    print
    print 'result translation: ', optimized_model.translation
    print 'result rotation: ', optimized_model.deg_rotation


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.pass_context
def edjust(ctx, input_folder):
    """Adjust pose of 3D object"""

    image, scene = load_input(input_folder)
    model_queue = SimpleQueue()
    process = Process(target=run_optimization,
                      args=(image, scene, model_queue))
    process.start()
    exit_code = run_gui(sys.argv[:1], image, scene, model_queue)
    process.terminate()
    ctx.exit(exit_code)


if __name__ == '__main__':
    edjust()  # pylint: disable=no-value-for-parameter
