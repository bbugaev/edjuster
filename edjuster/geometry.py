from collections import defaultdict, namedtuple

import numpy as np
import pyassimp
from pyassimp.postprocess import aiProcess_JoinIdenticalVertices
from pyassimp.postprocess import aiProcess_Triangulate


Mesh = namedtuple('Mesh', ('vertices', 'faces'))
Scene = namedtuple('Scene', ('mesh', 'model', 'view', 'proj'))
MeshEdges = namedtuple('MeshEdges',
                       ('projected_vertices', 'borders', 'sharp_edges'))


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


def _get_projected_vertices(scene):
    """Return projected vertices of given scene in homogeneous coordinates"""
    mvp = scene.proj.dot(scene.view.dot(scene.model))
    vertices = np.insert(scene.mesh.vertices, 3, 1, axis=1)
    vertices = np.dot(vertices, mvp.T)
    vertices /= vertices[:, -1:]
    vertices = np.insert(vertices[:, :2], 2, 1, axis=1)
    return vertices


def _calc_faces_of_edges(mesh):
    faces = mesh.faces
    faces.shape = faces.shape + (1,)
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


def detect_mesh_edges(scene):
    edge_faces = _calc_faces_of_edges(scene.mesh)
    vertices = _get_projected_vertices(scene)

    borders = []
    sharp_edges = []

    for edge, face_indices in edge_faces.iteritems():
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

    return MeshEdges(
        vertices[:, :-1],
        np.array(borders),
        np.array(sharp_edges)
    )
