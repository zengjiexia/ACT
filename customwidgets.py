from PySide6 import QtCore, QtGui, QtWidgets

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
