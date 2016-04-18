import numpy as np
from PySide import QtCore, QtGui, QtOpenGL
from OpenGL import GL

from geometry import detect_mesh_edges


WINDOW_TITLE = 'Edjuster'


def _create_texture_qimage(image):
    image_formats = {
        1: QtGui.QImage.Format_Indexed8,
        3: QtGui.QImage.Format_RGB888
    }

    channel_count = 1 if len(image.shape) == 2 else image.shape[2]
    if image.dtype != np.uint8 or channel_count not in image_formats:
        raise ValueError('Argument must be 8-bit grayscale or RGB888 image')

    image = np.flipud(image).copy()
    return QtGui.QImage(image.data, image.shape[1], image.shape[0],
                        image_formats[channel_count]).copy()


class Drawer(QtOpenGL.QGLWidget):
    WIREFRAME_COLOR = np.array([0.0, 1.0, 0.0])
    BORDER_COLOR = np.array([1.0, 0.0, 1.0])
    SHARP_EDGE_COLOR = np.array([1.0, 1.0, 0.0])
    POINT_COLOR = np.array([1.0, 0, 0])

    def __init__(self, image, scene, model_queue):
        QtOpenGL.QGLWidget.__init__(self)
        self.setWindowTitle(self.tr(WINDOW_TITLE))

        self.startTimer(40)

        self._call_list = []
        self._texture_id = 0
        self._image = _create_texture_qimage(image)
        self._scene = scene
        self._model_queue = model_queue
        self._update_mesh_edges()

    def timerEvent(self, _):
        new_model = None
        while not self._model_queue.empty():
            new_model = self._model_queue.get()
        if not new_model:
            return
        self._scene = self._scene._replace(model=new_model)
        self._update_mesh_edges()
        self.update()

    def initializeGL(self):
        Drawer._load_ortho()

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

        Drawer._push_matrices()
        self._load_scene_matrices()
        self._draw_mesh()
        self._draw_edges(self._mesh_edges.borders, Drawer.BORDER_COLOR)
        self._draw_edges(self._mesh_edges.sharp_edges, Drawer.SHARP_EDGE_COLOR)
        Drawer._pop_matrices()

    def _update_mesh_edges(self):
        self._mesh_edges = detect_mesh_edges(
            self._scene,
            (self._image.height(), self._image.width())
        )

    def _load_scene_matrices(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixd(self._scene.proj.T)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        model_view = self._scene.view.matrix.dot(self._scene.model.matrix).T
        GL.glLoadMatrixd(model_view)

    def _draw_image(self):
        self.qglColor(QtCore.Qt.white)
        self.drawTexture(QtCore.QRectF(-1, -1, 2, 2), self._texture_id)

    def _draw_mesh(self):
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

    @staticmethod
    def _load_ortho():
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1, 1, -1, 1, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    @staticmethod
    def _push_matrices():
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()

    @staticmethod
    def _pop_matrices():
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


def run_gui(argv, image, scene, model_queue):
    app = QtGui.QApplication(argv)

    if not QtOpenGL.QGLFormat.hasOpenGL():
        QtGui.QMessageBox.critical(None, WINDOW_TITLE,
                                   'This system does not support OpenGL')
        return 1

    window = Drawer(image, scene, model_queue)
    window.resize(800, 600)
    window.show()

    return app.exec_()
