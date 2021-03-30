import sys
import os

import PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import QFile, QIODevice, Slot
from PySide6.QtUiTools import QUiLoader
import pyqtgraph as pg
import customwidgets
from logics import SimPullAnalysis

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
                    return customwidgets.LogTextEdit(parent=parent) # promote to self defined LogTextEdit(QPlainTextEdit)
                return super().createWidget(className, parent, name)

        loader = UiLoader()
        self.window = loader.load(ui_file, self)

        # main window widgets
        self.window.main_runButton.clicked.connect(self.runAnalysis)



        ui_file.close()
        if not self.window:
            print(loader.errorString())
            sys.exit(-1)
        self.window.show()
        sys.exit(app.exec_())


    def update_log(self, message):
        self.window.main_logBar.insertPlainText(message)
        self.window.main_logBar.verticalScrollBar().setValue(
            self.window.main_logBar.verticalScrollBar().maximum())
        # auto scroll down to the newest message


    @Slot()
    def runAnalysis(self):
        data_path = self.window.main_pathEntry.toPlainText()
        self.update_log(data_path+'\n')


if __name__ == "__main__":
    app = QApplication([])
    widget = SimPullAnalysis_UI()
    widget.show()
    sys.exit(app.exec_())
