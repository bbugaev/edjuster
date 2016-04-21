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
from optimization import IntegralCalculator, optimize_model


def load_file(folder, filename, hint, loader):
    filename = path.join(folder, filename)
    try:
        result = loader(filename)
    except IOError:
        raise click.FileError(filename, hint)
    return result


def load_input(input_folder):
    rgb = load_file(input_folder, 'image.png', '3D object image',
                    partial(imread, mode='RGB'))
    gray = load_file(input_folder, 'image.png', '3D object image',
                     partial(imread, mode='F'))
    mesh = load_file(input_folder, 'mesh.obj', '3D object', load_mesh)
    proj = load_file(input_folder, 'proj.txt', 'proj matrix', np.loadtxt)

    model = load_file(input_folder, 'model.txt', 'model matrix', np.loadtxt)
    model = Position(model.flatten())

    view = load_file(input_folder, 'view.txt', 'view matrix', np.loadtxt)
    view = Position(view.flatten())

    return rgb, gray / 255.0, Scene(mesh, model, view, proj)


def run_optimization(model, integral_calculator, model_queue):

    def process_step(vector6, _, accepted):
        if accepted:
            model_queue.put(Position(vector6))

    optimized_model = optimize_model(model, integral_calculator,
                                     process_step, True)
    model_queue.put(optimized_model)

    print
    print 'result translation: ', optimized_model.translation
    print 'result rotation: ', optimized_model.deg_rotation


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.pass_context
def edjust(ctx, input_folder):
    """Adjust pose of 3D object"""

    rgb, gray, scene = load_input(input_folder)
    integral_calculator = IntegralCalculator(gray, scene)
    model_queue = SimpleQueue()
    process = Process(target=run_optimization,
                      args=(scene.model, integral_calculator, model_queue))
    process.start()
    exit_code = run_gui(sys.argv[:1], rgb, scene, integral_calculator,
                        model_queue)
    process.terminate()
    ctx.exit(exit_code)


if __name__ == '__main__':
    edjust()  # pylint: disable=no-value-for-parameter
