from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, Signal, Slot, QAbstractTableModel, Qt
import pandas as pd
import sys
import imagej
import os

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


class PandasModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class DFLParticleFinder(QObject):
    finished = Signal()
    progress = Signal(int)

    def __init__(self, algorithm, project, size, threshold, IJ):
        super().__init__()

        self.algorithm = algorithm
        self.project = project
        self.size = size
        self.threshold = threshold
        self.IJ = IJ

    @QtCore.Slot()
    def run(self):
        if self.algorithm == 'ComDet':
            try:
                self.project.call_ComDet(self.size, self.threshold, progress_signal=self.progress, IJ=self.IJ)
            except:
                print(sys.exc_info())
        elif self.algorithm == 'Trevor':
            try:
                self.project.call_Trevor(self.size, self.threshold, progress_signal=self.progress)
            except:
                print(sys.exc_info())
        else:
            pass

        self.finished.emit()


class ReportWriter(QObject):
    finished = Signal()
    progress = Signal(int)

    def __init__(self, project):
        super().__init__()
        self.project = project


    @QtCore.Slot()
    def run(self):
        try:
            self.project.generate_reports(progress_signal=self.progress)
        except:
            print(sys.exc_info())

        self.finished.emit()


class LipoAssayWorker(QObject):
    finished = Signal()
    progress = Signal(int)
    log = Signal(str)
    
    def __init__(self, project, threshold):
        super().__init__()
        self.project = project
        self.threshold = threshold


    @QtCore.Slot()
    def run(self):
        try:
            self.project.run_analysis(threshold=self.threshold, progress_signal=self.progress, log_signal=self.log)
        except:
            print(sys.exc_info())

        self.finished.emit()


class SRWorker(QObject):
    finished = Signal()
    progress = Signal(int)

    def __init__(self, job, project, IJ):
        super().__init__()

        self.job = job
        self.project = project
        self.IJ = IJ

    @QtCore.Slot()
    def run(self):
        if self.job == 'Reconstruction':
            try:
                self.project.superRes_reconstruction(progress_signal=self.progress, IJ=self.IJ)
            except:
                print(sys.exc_info())
        elif self.job == 'FiducialCorrection':
            try:
                self.project.superRes_fiducialCorrection(progress_signal=self.progress, IJ=self.IJ)
            except:
                print(sys.exc_info())

        self.finished.emit()