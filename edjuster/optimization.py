from itertools import izip

import numpy as np


def select_points(mesh_edges, point_count):
    edges = np.vstack((mesh_edges.borders, mesh_edges.sharp_edges))
    vertices = np.hsplit(mesh_edges.projected_vertices[edges], 2)

    lengths = np.linalg.norm(vertices[1] - vertices[0], axis=2)
    lengths.shape = lengths.shape[:1]
    total_length = lengths.sum()
    step = total_length / (point_count + 1)

    vertices = np.hstack(vertices)

    points = []
    current_length = step
    prefix_length = 0

    for length, line in izip(lengths, vertices):
        while current_length <= prefix_length + length:
            point_coef = (current_length - prefix_length) / length
            points.append(line[0] + (line[1] - line[0]) * point_coef)
            current_length += step
        prefix_length += length

    return np.array(points[:point_count])
