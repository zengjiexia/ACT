from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, Signal, Slot
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



class FijiWorker(QObject):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, IJ, project, size, threshold):
        super().__init__()

        self.IJ = IJ
        self.project = project
        self.size = size
        self.threshold = threshold

    @QtCore.Slot()
    def run(self):

        def extract_FoV(path):
            """
            #get the name of field of views for a sample (format - XnYnRnWnCn)
            #para: path - string
            #return: fov_path - dict[fov] = path
            """
            fov_path = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.tif'):
                        fov_path[file[:10]] = os.path.join(root, file)
            return fov_path

        fov_paths = extract_FoV(self.project.path_data_main)

        for field in sorted(fov_paths):
            imgFile = fov_paths[field]
            saveto = os.path.join(self.project.path_result_raw, field)
            saveto = saveto.replace("\\", "/")
            img = self.IJ.io().open(imgFile)
            self.IJ.ui().show(field, img)
            macro = """
            run("Z Project...", "projection=[Average Intensity]");
            run("Detect Particles", "ch1i ch1a="""+str(self.size)+""" ch1s="""+str(self.threshold)+""" rois=Ovals add=Nothing summary=Reset");
            selectWindow('Results');
            saveAs("Results", \""""+saveto+"""_results.csv\");
            close("Results");
            selectWindow('Summary');
            saveAs("Results", \""""+saveto+"""_summary.txt\");
            close(\""""+field+"""_summary.txt\");
            selectWindow(\"AVG_"""+field+"""\");
            saveAs("tif", \""""+saveto+""".tif\");
            close();
            close();
            """
            try:
                self.IJ.py.run_macro(macro)
            except:
                print(sys.exc_info())
        self.finished.emit()
