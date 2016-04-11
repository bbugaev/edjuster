from collections import namedtuple

import numpy as np
import pyassimp


Mesh = namedtuple('Mesh', ('vertices', 'faces'))


def _make_vector_array(vector):
    return [vector.x, vector.y, vector.z]


def _make_face_array(face):
    return [face.mIndices[i] for i in xrange(face.mNumIndices)]


def load_mesh(filename):
    try:
        scene = pyassimp.load(filename)
    except pyassimp.AssimpError as error:
        raise IOError(error.message)

    mesh = scene.mMeshes[0].contents
    vertices = np.array([_make_vector_array(mesh.mVertices[i])
                         for i in xrange(mesh.mNumVertices)])
    faces = np.array([_make_face_array(mesh.mFaces[i])
                      for i in xrange(mesh.mNumFaces)])

    pyassimp.release(scene)

    return Mesh(vertices, faces)
