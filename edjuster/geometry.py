
import numpy as np
import pyassimp
from pyassimp.postprocess import aiProcess_JoinIdenticalVertices
from pyassimp.postprocess import aiProcess_Triangulate


Mesh = namedtuple('Mesh', ('vertices', 'faces'))
Scene = namedtuple('Scene', ('mesh', 'model', 'view', 'proj'))


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
