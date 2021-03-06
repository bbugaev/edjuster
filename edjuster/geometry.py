from collections import defaultdict, namedtuple

import numpy as np
import pyassimp
from pyassimp.postprocess import aiProcess_JoinIdenticalVertices
from pyassimp.postprocess import aiProcess_Triangulate


Mesh = namedtuple('Mesh', ('vertices', 'faces'))
Scene = namedtuple('Scene', ('mesh', 'model', 'view', 'proj'))
BBox = namedtuple('BBox',
                  ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'))


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


def calc_bbox(mesh):
    x_coords = mesh.vertices[:, 0]
    y_coords = mesh.vertices[:, 1]
    z_coords = mesh.vertices[:, 2]
    return BBox(x_coords.min(), x_coords.max(),
                y_coords.min(), y_coords.max(),
                z_coords.min(), z_coords.max())


def convert_to_format(points, image_size):
    points = points + np.array([1, 1])
    points *= np.array([image_size[1], image_size[0]]) / 2.0
    return points


def convert_from_format(points, image_size):
    points = points / (np.array([image_size[1], image_size[0]]) / 2.0)
    points -= np.array([1, 1])
    return points


class MeshEdgeDetector(object):
    MeshEdges = namedtuple('MeshEdges',
                           ('projected_vertices', 'borders', 'sharp_edges'))

    def __init__(self, mesh):
        self._mesh = mesh
        self._edges, self._faces = MeshEdgeDetector._find_faces_of_edges(mesh)

    def __call__(self, mvp, image_size):
        vertices = self._get_projected_vertices(mvp, image_size)

        edges, faces = self._edges, self._faces
        front_masks = [MeshEdgeDetector._are_front(vertices[f]) for f in faces]

        borders = edges[front_masks[0] != front_masks[1]]

        front_masks_and = front_masks[0] & front_masks[1]
        edges = edges[front_masks_and]
        faces = [f[front_masks_and] for f in faces]
        normals = [MeshEdgeDetector._calc_normals(self._mesh.vertices[f])
                   for f in faces]
        angles = np.arccos((normals[0] * normals[1]).sum(axis=1))
        sharp_edges = edges[angles >= np.pi / 2]

        return MeshEdgeDetector.MeshEdges(vertices, borders, sharp_edges)

    def _get_projected_vertices(self, mvp, image_size):
        vertices = np.insert(self._mesh.vertices, 3, 1, axis=1)
        vertices = np.dot(vertices, mvp.T)
        vertices /= vertices[:, -1:]
        vertices = convert_to_format(vertices[:, :2], image_size)
        return vertices

    @staticmethod
    def _find_faces_of_edges(mesh):
        faces = mesh.faces[:, :, np.newaxis]
        rolled_faces = np.roll(faces, 1, axis=1)
        edges = np.dstack((faces, rolled_faces))
        edges.sort()

        edge_faces = defaultdict(list)
        for face_idx, face_edges in enumerate(edges):
            for edge in face_edges:
                edge_faces[tuple(edge)].append(face_idx)

        edge_faces = {e: f for e, f in edge_faces.iteritems() if len(f) == 2}
        result_edges = np.array(edge_faces.keys())
        result_faces = mesh.faces[np.array(edge_faces.values())]
        result_faces = [f.reshape(-1, 3) for f in np.hsplit(result_faces, 2)]
        return result_edges, result_faces

    @staticmethod
    def _cross(triangles):
        points = [p.reshape(-1, triangles.shape[-1])
                  for p in np.hsplit(triangles, 3)]
        return np.cross(points[1] - points[0], points[2] - points[0])

    @staticmethod
    def _are_front(triangles):
        return MeshEdgeDetector._cross(triangles) > 0

    @staticmethod
    def _calc_normals(triangles):
        normals = MeshEdgeDetector._cross(triangles)
        norms = np.repeat(np.linalg.norm(normals, axis=1), 3).reshape(-1, 3)
        return normals / norms
