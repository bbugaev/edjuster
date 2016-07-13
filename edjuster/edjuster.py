#! /usr/bin/env python2

from multiprocessing import Process
from multiprocessing.queues import SimpleQueue
import sys

import click

from geometry import Position
from gui import run_gui
from optimization import IntegralCalculator, optimize_model
from reading import read_input


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

    rgb, gray, scene = read_input(input_folder)
    integral_calculator = IntegralCalculator(gray, scene,
                                             normalized_gradient=True)
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
