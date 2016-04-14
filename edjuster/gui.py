import numpy as np
from PySide import QtCore, QtGui, QtOpenGL
from OpenGL import GL


WINDOW_TITLE = 'Edjuster'


def _create_qimage(image):
    image_formats = {
        1: QtGui.QImage.Format_Indexed8,
        3: QtGui.QImage.Format_RGB888
    }

    channel_count = 1 if len(image.shape) == 2 else image.shape[2]
    if image.dtype != np.uint8 or channel_count not in image_formats:
        raise ValueError('Argument must be 8-bit grayscale or RGB888 image')

    return QtGui.QImage(image.data, image.shape[1], image.shape[0],
                        image_formats[channel_count])


class Drawer(QtOpenGL.QGLWidget):
    WIREFRAME_COLOR = np.array([0.0, 1.0, 0.0])
    BORDER_COLOR = np.array([1.0, 0.0, 1.0])
    SHARP_EDGE_COLOR = np.array([1.0, 1.0, 0.0])

    def __init__(self, image, scene, mesh_edges):
        QtOpenGL.QGLWidget.__init__(self)
        self.setWindowTitle(self.tr(WINDOW_TITLE))

        self._call_list = []
        self._texture_id = 0
        self._image = _create_qimage(image)
        self._scene = scene
        self._mesh_edges = mesh_edges

    def initializeGL(self):
        self._texture_id = self.bindTexture(self._image)
        self._init_call_list()

        GL.glClearColor(0, 0, 0, 0)
        GL.glClearDepth(1)
        GL.glEnable(GL.GL_BLEND)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glFrontFace(GL.GL_CCW)
        GL.glCullFace(GL.GL_BACK)

    def resizeGL(self, window_width, window_height):
        coef = min(1.0,
                   float(window_width) / self._image.width(),
                   float(window_height) / self._image.height())

        viewport_width = int(coef * self._image.width())
        viewport_height = int(coef * self._image.height())

        GL.glViewport((window_width - viewport_width) / 2,
                      (window_height - viewport_height) / 2,
                      viewport_width,
                      viewport_height)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._draw_image()
        self._draw_mesh()

    def _draw_image(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.5, 0.5, 0.5, -0.5, -0.5, 0.5)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        self.qglColor(QtCore.Qt.white)
        self.drawTexture(QtCore.QRectF(-0.5, -0.5, 1, 1), self._texture_id)

    def _draw_mesh(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixd(self._scene.proj.T)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        model_view = self._scene.view.dot(self._scene.model).T
        GL.glLoadMatrixd(model_view)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1, 1)
        GL.glColor4d(1, 1, 1, 1)
        GL.glCallList(self._call_list)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        GL.glDepthMask(GL.GL_FALSE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glColor3dv(Drawer.WIREFRAME_COLOR)
        GL.glCallList(self._call_list)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        self._draw_edges(self._mesh_edges.borders, Drawer.BORDER_COLOR)
        self._draw_edges(self._mesh_edges.sharp_edges, Drawer.SHARP_EDGE_COLOR)

    def _draw_edges(self, edges, color):
        GL.glColor3dv(color)
        GL.glBegin(GL.GL_LINES)
        for line in edges:
            for vertex in self._scene.mesh.vertices[line]:
                GL.glVertex3dv(vertex)
        GL.glEnd()

    def _init_call_list(self):
        self._call_list = GL.glGenLists(1)
        GL.glNewList(self._call_list, GL.GL_COMPILE)

        for face in self._scene.mesh.faces:
            GL.glBegin(GL.GL_POLYGON)
            for vertex in self._scene.mesh.vertices[face]:
                GL.glVertex3dv(vertex)
            GL.glEnd()

        GL.glEndList()


def run_gui(argv, image, scene, mesh_edges):
    app = QtGui.QApplication(argv)

    if not QtOpenGL.QGLFormat.hasOpenGL():
        QtGui.QMessageBox.critical(None, WINDOW_TITLE,
                                   'This system does not support OpenGL')
        return 1

    window = Drawer(image, scene, mesh_edges)
    window.resize(800, 600)
    window.show()

    return app.exec_()
