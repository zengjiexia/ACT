import sys
import os

import PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QProgressDialog
from PySide6.QtCore import QFile, QIODevice, Slot, Qt, QThread, QObject, pyqtSignal
from PySide6.QtUiTools import QUiLoader
import pyqtgraph as pg
import toolbox
from logics import SimPullAnalysis, Worker

class SimPullAnalysis_UI(QMainWindow):

    def __init__(self):
        super(SimPullAnalysis_UI, self).__init__()
        self.load_ui()

    def load_ui(self):

        path = os.path.join(os.path.dirname(__file__), "form.ui")
        ui_file = QFile(path)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)

        class UiLoader(QUiLoader):
            def createWidget(self, className, parent=None, name=""):
                if className == "PlotWidget":
                    return pg.PlotWidget(parent=parent) # promote to pyqtgraph.PlotWidget
                if className == "LogTextEdit":
                    return toolbox.LogTextEdit(parent=parent) # promote to self defined LogTextEdit(QPlainTextEdit)
                return super().createWidget(className, parent, name)

        loader = UiLoader()
        self.window = loader.load(ui_file, self)

        # main window widgets
        self.window.main_runButton.clicked.connect(self.clickMainWindowRun)



        ui_file.close()
        if not self.window:
            print(loader.errorString())
            sys.exit(-1)
        self.window.show()
        sys.exit(app.exec_())


    def update_log(self, message):
        self.window.main_logBar.insertPlainText(message + '\n')
        self.window.main_logBar.verticalScrollBar().setValue(
            self.window.main_logBar.verticalScrollBar().maximum())
        # auto scroll down to the newest message


    def show_message(self, msg_type, message):
        msgBox = QMessageBox(self.window)
        if msg_type == 'c':
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setWindowTitle('Critical Error')
        elif msg_type == 'w':
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle('Warning')
        msgBox.setText(message)
        returnValue = msgBox.exec_()


    def clickMainWindowRun(self):
        self._checkParameters()
        self._runAnalysis()
        self.update_log('Particles in images are located.')


    def _checkParameters(self):
        #Check if data path exist
        data_path = self.window.main_pathEntry.toPlainText()
        if os.path.isdir(data_path) == False:
            self.show_message('w', 'Error: Path to folder not found.')
        else:
            self.data_path = data_path
            self.window.main_pathEntry.setPlainText(self.data_path)
            self.update_log('Data path set to '+data_path)

        #Check input threshold
        try:
            self.threshold = int(self.window.main_thresholdEntry.toPlainText())
            self.window.main_thresholdEntry.setPlainText(str(self.threshold))
        except TypeError:
            self.show_message('c', 'Please input a number for threshold.')
        if self.threshold <= 2 or self.threshold >= 20:
            self.update_log('Threshold set as '+str(self.threshold)+' SD. Suggested range would be 3-20 SD.')
        else:
            self.update_log('Threshold set at '+str(self.threshold)+' SD.')

        #check input estimated size
        try:
            self.size = int(self.window.main_sizeEntry.toPlainText())
            self.window.main_sizeEntry.setPlainText(str(self.size))
        except TypeError:
            self.show_message('c', 'Please input a number for estimated particle size.')
        if self.size >= 15:
            self.update_log('Estimated particle size set as '+str(self.size)+' pixels which is quite high. Pariticles close to each other might be considered as one.')
        else:
            self.update_log('Estimated particle size set as '+str(self.size)+' pixels.')


    def _runAnalysis(self):
        self.project = SimPullAnalysis(self.data_path)


        progress.setWindowModality(Qt.WindowModal)
        self.project.call_ComDet_UI(self.size, self.threshold)



if __name__ == "__main__":

    app = QApplication([])
    #app.setQuitOnLastWindowClosed(False)
    widget = SimPullAnalysis_UI()
    widget.show()
    sys.exit(app.exec_())
