from PySide import QtCore, QtGui, QtOpenGL
from OpenGL import GL


WINDOW_TITLE = 'Edjuster'


class Drawer(QtOpenGL.QGLWidget):

    def __init__(self):
        QtOpenGL.QGLWidget.__init__(self)
        self.setWindowTitle(self.tr(WINDOW_TITLE))

        self._call_list = []

    def initializeGL(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.5, 0.5, 0.5, -0.5, -0.5, 0.5)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)

        self._init_call_list()

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
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


def run_gui(argv):
    app = QtGui.QApplication(argv)

    if not QtOpenGL.QGLFormat.hasOpenGL():
        QtGui.QMessageBox.critical(None, WINDOW_TITLE,
                                   'This system does not support OpenGL')
        return 1

    window = Drawer()
    window.resize(800, 600)
    window.show()

    return app.exec_()
