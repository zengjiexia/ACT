import sys
import os

import PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtUiTools import QUiLoader
import pyqtgraph as pg
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
                    return pg.PlotWidget(parent=parent)
                return super().createWidget(className, parent, name)

        loader = UiLoader()
        window = loader.load(ui_file, self)
        ui_file.close()
        if not window:
            print(loader.errorString())
            sys.exit(-1)
        window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication([])
    widget = SimPullAnalysis_UI()
    widget.show()
    sys.exit(app.exec_())
