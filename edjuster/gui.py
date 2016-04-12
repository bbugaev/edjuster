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

    def __init__(self, image):
        QtOpenGL.QGLWidget.__init__(self)
        self.setWindowTitle(self.tr(WINDOW_TITLE))

        self._call_list = []
        self._texture_id = 0
        self._image = _create_qimage(image)

    def initializeGL(self):
        GL.glClearColor(0, 0, 0, 0)
        self._texture_id = self.bindTexture(self._image)
        self._init_call_list()

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
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self._draw_texture()
        GL.glCallList(self._call_list)

    def _init_call_list(self):
        self._call_list = GL.glGenLists(1)
        GL.glNewList(self._call_list, GL.GL_COMPILE)

        self.qglColor(QtCore.Qt.green)
        GL.glBegin(GL.GL_POLYGON)
        GL.glVertex2d(0.0, 0.0)
        GL.glVertex2d(0.0, 0.5)
        GL.glVertex2d(0.5, 0.5)
        GL.glVertex2d(0.5, 0.0)
        GL.glEnd()

        GL.glEndList()

    def _draw_texture(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.5, 0.5, 0.5, -0.5, -0.5, 0.5)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        self.qglColor(QtCore.Qt.white)
        self.drawTexture(QtCore.QRectF(-0.5, -0.5, 1, 1), self._texture_id)


def run_gui(argv, image):
    app = QtGui.QApplication(argv)

    if not QtOpenGL.QGLFormat.hasOpenGL():
        QtGui.QMessageBox.critical(None, WINDOW_TITLE,
                                   'This system does not support OpenGL')
        return 1

    window = Drawer(image)
    window.resize(800, 600)
    window.show()

    return app.exec_()
