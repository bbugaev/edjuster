#! /usr/bin/env python2

import click


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
def edjust(input_folder):
    """Adjusts pose of 3D object"""
    click.echo('INPUT_FOLDER: %s' % input_folder)


if __name__ == '__main__':
    edjust() # pylint: disable=no-value-for-parameter
