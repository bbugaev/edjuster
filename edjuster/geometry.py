from collections import defaultdict, namedtuple

import numpy as np
import pyassimp
from pyassimp.postprocess import aiProcess_JoinIdenticalVertices
from pyassimp.postprocess import aiProcess_Triangulate


Mesh = namedtuple('Mesh', ('vertices', 'faces'))
Scene = namedtuple('Scene', ('mesh', 'model', 'view', 'proj'))
MeshEdges = namedtuple('MeshEdges',
                       ('projected_vertices', 'borders', 'sharp_edges'))


class Position(object):
    DEG = 'deg'
    RAD = 'rad'

    def __init__(self, vector6, rotation_unit=DEG):
        if rotation_unit not in (Position.DEG, Position.RAD):
            raise ValueError(
                'Invalid rotation unit (use Position.DEG or Position.RAD)'
            )
        self.vector6 = np.array(vector6, dtype='float64', copy=True)
        if rotation_unit == Position.RAD:
            vector6[3:] *= 180.0 / np.pi

    @property
    def translation(self):
        return self.vector6[:3]

    @property
    def deg_rotation(self):
        return self.vector6[3:]

    @property
    def rad_rotation(self):
        return self.deg_rotation * np.pi / 180.0

    @property
    def matrix(self):
        sines = np.sin(self.rad_rotation)
        cosines = np.cos(self.rad_rotation)
        x_rotation = np.array([[1, 0, 0],
                               [0, cosines[0], -sines[0]],
                               [0, sines[0], cosines[0]]])
        y_rotation = np.array([[cosines[1], 0, sines[1]],
                               [0, 1, 0],
                               [-sines[1], 0, cosines[1]]])
        z_rotation = np.array([[cosines[2], -sines[2], 0],
                               [sines[2], cosines[2], 0],
                               [0, 0, 1]])
        result = np.eye(4)
        result[:3, 3] = self.translation
        result[0:3, 0:3] = z_rotation.dot(y_rotation.dot(x_rotation))
        return result


def _make_vector_array(vector):
    return [vector.x, vector.y, vector.z]


def _make_face_array(face):
    return [face.mIndices[i] for i in xrange(face.mNumIndices)]


def load_mesh(filename):
    """Load mesh from file and triangulate it"""

    try:
        scene = pyassimp.load(
            filename,
            processing=aiProcess_JoinIdenticalVertices | aiProcess_Triangulate
        )
    except pyassimp.AssimpError as error:
        raise IOError(error.message)

    mesh = scene.mMeshes[0].contents
    vertices = np.array([_make_vector_array(mesh.mVertices[i])
                         for i in xrange(mesh.mNumVertices)])
    faces = np.array([_make_face_array(mesh.mFaces[i])
                      for i in xrange(mesh.mNumFaces)])

    pyassimp.release(scene)

    return Mesh(vertices, faces)


def convert_to_format(points, image_size):
    points = points + np.array([1, 1])
    points *= np.array([image_size[1], image_size[0]]) / 2.0
    return points


def convert_from_format(points, image_size):
    points = points / (np.array([image_size[1], image_size[0]]) / 2.0)
    points -= np.array([1, 1])
    return points


def _get_projected_vertices(scene, image_size):
    """Return projected vertices of given scene in homogeneous coordinates"""
    mvp = scene.proj.dot(scene.view.matrix.dot(scene.model.matrix))
    vertices = np.insert(scene.mesh.vertices, 3, 1, axis=1)
    vertices = np.dot(vertices, mvp.T)
    vertices /= vertices[:, -1:]
    vertices = convert_to_format(vertices[:, :2], image_size)
    vertices = np.insert(vertices, 2, 1, axis=1)
    return vertices


def _calc_faces_of_edges(mesh):
    faces = mesh.faces[:, :, np.newaxis]
    rolled_faces = np.roll(faces, 1, axis=1)
    edges = np.dstack((faces, rolled_faces))
    edges.sort()

    edge_faces = defaultdict(list)
    for face_idx, face_edges in enumerate(edges):
        for edge in face_edges:
            edge_faces[tuple(edge)].append(face_idx)

    return edge_faces


def _is_front(triangle):
    return np.linalg.det(np.vstack(triangle)) > 0


def _calc_normal(triangle):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    return normal


def detect_mesh_edges(scene, image_size):
    edge_faces = _calc_faces_of_edges(scene.mesh)
    vertices = _get_projected_vertices(scene, image_size)

    borders = []
    sharp_edges = []

    for edge, face_indices in edge_faces.iteritems():
        if len(face_indices) != 2:
            continue

        face_1, face_2 = (scene.mesh.faces[i] for i in face_indices)
        front_1 = _is_front(vertices[face_1])
        front_2 = _is_front(vertices[face_2])

        if front_1 != front_2:
            borders.append(edge)

        if front_1 and front_2:
            normal_1 = _calc_normal(scene.mesh.vertices[face_1])
            normal_2 = _calc_normal(scene.mesh.vertices[face_2])
            angle_between_normals = np.math.acos(np.inner(normal_1, normal_2))

            if angle_between_normals >= np.math.pi / 2:
                sharp_edges.append(edge)

    vertices = vertices[:, :-1]
    borders = np.array(borders)
    if sharp_edges:
        sharp_edges = np.array(sharp_edges)
    else:
        sharp_edges = np.ndarray((0, 2), borders.dtype)
    return MeshEdges(vertices, borders, sharp_edges)
