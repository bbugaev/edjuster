#! /usr/bin/env python2

import os.path as path

import click
from scipy.ndimage import imread


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
def edjust(input_folder):
    """Adjusts pose of 3D object"""

    image_path = path.join(input_folder, 'image.bmp')
    try:
        image = imread(image_path)
    except IOError:
        raise click.FileError(image_path, '3D object image')


if __name__ == '__main__':
    edjust() # pylint: disable=no-value-for-parameter
