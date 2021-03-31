from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, pyqtSignal

class LogTextEdit(QtWidgets.QPlainTextEdit):
    def write(self, message):
        if not hasattr(self, "flag"):
            self.flag = False
        message = message.replace('\r', '').rstrip()
        if message:
            method = "replace_last_line" if self.flag else "appendPlainText"
            QtCore.QMetaObject.invokeMethod(self,
                method,
                QtCore.Qt.QueuedConnection, 
                QtCore.Q_ARG(str, message))
            self.flag = True
        else:
            self.flag = False

    @QtCore.Slot(str)
    def replace_last_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.select(QtGui.QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertBlock()
        self.setTextCursor(cursor)
        self.insertPlainText(text)


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, job):
        self.job = job
        
    def run(self):
        """Long-running task."""
        for i in range(5):
            sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()